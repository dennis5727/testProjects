import os
os.environ.setdefault("SECRET_KEY", "integration-test-secret-at-least-32-bytes")

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from fastapi_blog import models
from fastapi_blog.auth import hash_reset_token
from fastapi_blog.database import Base, get_db
from fastapi_blog.main import app


class UpdateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async def test_db():
            async with self.sessions() as session:
                yield session

        app.dependency_overrides[get_db] = test_db
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        )
        self.mail_patch = patch(
            "fastapi_blog.routers.users.send_password_reset_email", new_callable=AsyncMock
        )
        self.mail = self.mail_patch.start()
        response = await self.client.post("/api/users", json={
            "username": "tester", "email": "tester@example.com", "password": "old-password"
        })
        self.assertEqual(response.status_code, 201)
        self.user_id = response.json()["id"]
        response = await self.login("old-password")
        self.headers = {"Authorization": f"Bearer {response.json()['access_token']}"}

    async def asyncTearDown(self):
        self.mail_patch.stop()
        app.dependency_overrides.clear()
        await self.client.aclose()
        await self.engine.dispose()

    async def login(self, password):
        return await self.client.post("/api/users/token", data={
            "username": "tester@example.com", "password": password
        })

    async def request_reset(self):
        response = await self.client.post("/api/users/forgot-password", json={
            "email": "TESTER@example.com"
        })
        self.assertEqual(response.status_code, 202)
        return self.mail.call_args.kwargs["token"]

    async def reset(self, token):
        return await self.client.post("/api/users/reset-password", json={
            "token": token, "new_password": "new-password"
        })

    async def test_reset_is_hashed_single_use_and_changes_login(self):
        token = await self.request_reset()
        async with self.sessions() as session:
            row = (await session.execute(select(models.PasswordResetToken))).scalar_one()
            self.assertEqual(row.token_hash, hash_reset_token(token))
            self.assertNotEqual(row.token_hash, token)
        self.assertEqual((await self.reset(token)).status_code, 200)
        self.assertEqual((await self.reset(token)).status_code, 400)
        self.assertEqual((await self.login("old-password")).status_code, 401)
        self.assertEqual((await self.login("new-password")).status_code, 200)

    async def test_reset_request_replaces_previous_token_and_hides_unknown_email(self):
        old = await self.request_reset()
        new = await self.request_reset()
        self.assertEqual((await self.reset(old)).status_code, 400)
        self.assertEqual((await self.reset(new)).status_code, 200)
        count = self.mail.await_count
        response = await self.client.post("/api/users/forgot-password", json={
            "email": "missing@example.com"
        })
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.mail.await_count, count)

    async def test_expired_invalid_and_short_password(self):
        token = await self.request_reset()
        async with self.sessions() as session:
            row = (await session.execute(select(models.PasswordResetToken))).scalar_one()
            row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
            await session.commit()
        self.assertEqual((await self.reset(token)).status_code, 400)
        self.assertEqual((await self.reset("invalid")).status_code, 400)
        response = await self.client.post("/api/users/reset-password", json={
            "token": token, "new_password": "short"
        })
        self.assertEqual(response.status_code, 422)

    async def test_change_password_requires_auth_and_invalidates_reset(self):
        token = await self.request_reset()
        payload = {"current_password": "old-password", "new_password": "changed-password"}
        response = await self.client.patch("/api/users/me/password", json=payload)
        self.assertEqual(response.status_code, 401)
        response = await self.client.patch("/api/users/me/password", headers=self.headers,
            json={**payload, "current_password": "wrong-password"})
        self.assertEqual(response.status_code, 400)
        response = await self.client.patch("/api/users/me/password", headers=self.headers, json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await self.reset(token)).status_code, 400)
        self.assertEqual((await self.login("changed-password")).status_code, 200)

    async def test_pagination_pages_and_ownership(self):
        for i in range(12):
            response = await self.client.post("/api/posts", headers=self.headers,
                json={"title": f"Post {i}", "content": "Content"})
            self.assertEqual(response.status_code, 201)
        for path in ("/api/posts", f"/api/users/{self.user_id}/posts"):
            first = (await self.client.get(path, params={"limit": 10})).json()
            last = (await self.client.get(path, params={"skip": 10, "limit": 10})).json()
            self.assertEqual(first["total"], 12)
            self.assertEqual(len(first["posts"]), 10)
            self.assertTrue(first["has_more"])
            self.assertEqual(len(last["posts"]), 2)
            self.assertFalse(last["has_more"])
            self.assertFalse({p["id"] for p in first["posts"]} & {p["id"] for p in last["posts"]})
            self.assertEqual((await self.client.get(path, params={"skip": -1})).status_code, 422)
            self.assertEqual((await self.client.get(path, params={"limit": 101})).status_code, 422)
        for path in ("/", f"/users/{self.user_id}/posts", "/account", "/login", "/forgot-password", "/reset-password"):
            response = await self.client.get(path)
            self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        post_id = first["posts"][0]["id"]
        page = await self.client.get(f"/posts/{post_id}")
        self.assertIn(f"const postId = {post_id};", page.text)
        self.assertIn(f"const postUserId = {self.user_id};", page.text)
        response = await self.client.patch(f"/api/posts/{post_id}", headers=self.headers, json={"title": "Updated"})
        self.assertEqual(response.status_code, 200)

    async def test_email_template_paths_and_content(self):
        from fastapi_blog.email_utils import send_password_reset_email
        with patch("fastapi_blog.email_utils.send_email", new_callable=AsyncMock) as send:
            await send_password_reset_email("tester@example.com", "<tester>", "sample-token")
            content = send.call_args.kwargs
            self.assertIn("/reset-password?token=sample-token", content["plain_text"])
            self.assertIn("&lt;tester&gt;", content["html_content"])
            self.assertIn("60 minutes", content["html_content"])

    async def test_delete_user_cascades_reset_tokens(self):
        await self.request_reset()
        response = await self.client.delete(f"/api/users/{self.user_id}", headers=self.headers)
        self.assertEqual(response.status_code, 204)
        async with self.sessions() as session:
            self.assertIsNone((await session.execute(select(models.PasswordResetToken))).scalar_one_or_none())


if __name__ == "__main__":
    unittest.main()
