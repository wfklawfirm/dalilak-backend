"""
email_service.py — Transactional email via Resend API.

SECURITY: RESEND_API_KEY is read from the environment only. It is NEVER
logged, printed, committed, or included in any error message, test fixture,
or response body.

Usage
-----
    from email_service import send_reset_email
    sent = await send_reset_email(to_email, reset_url)
    # sent is True on success, False on any failure.
    # Callers must always return the same safe response regardless of `sent`
    # to prevent email-enumeration attacks.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_RESEND_URL = "https://api.resend.com/emails"


async def send_reset_email(
    to_email: str,
    reset_url: str,
    *,
    from_email: str | None = None,
) -> bool:
    """
    Send a password-reset link to `to_email` via Resend.

    Returns True on success, False on any failure.
    Logs a warning (not an error) on failure so callers can keep serving.

    Parameters
    ----------
    to_email   : recipient address
    reset_url  : full URL containing the raw (unhashed) reset token
    from_email : sender address; falls back to RESEND_FROM_EMAIL env var
                 then to "noreply@dalilak.ai"
    """
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "[email_service] RESEND_API_KEY not configured — password-reset "
            "email was NOT sent. Set this env var in Render to enable email."
        )
        return False

    sender = from_email or os.environ.get("RESEND_FROM_EMAIL", "noreply@dalilak.ai")

    body_html = f"""
    <div dir="rtl" style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;
                          padding:24px;background:#f9fafb;border-radius:8px;">
      <h2 style="color:#1a1a2e;margin-bottom:8px;">دليلك AI</h2>
      <p style="color:#374151;font-size:16px;">
        لقد طلبت إعادة تعيين كلمة المرور لحسابك. انقر على الزر أدناه لإكمال العملية:
      </p>
      <p style="text-align:center;margin:32px 0;">
        <a href="{reset_url}"
           style="display:inline-block;padding:14px 28px;background:#4f46e5;
                  color:#ffffff;text-decoration:none;border-radius:6px;
                  font-weight:bold;font-size:15px;">
          إعادة تعيين كلمة المرور
        </a>
      </p>
      <p style="color:#6b7280;font-size:13px;line-height:1.6;">
        ⏱ هذا الرابط صالح لمدة <strong>15 دقيقة</strong> فقط وللاستخدام مرة واحدة.<br>
        إذا لم تطلب إعادة التعيين، يمكنك تجاهل هذه الرسالة بأمان.
      </p>
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;">
      <p style="color:#9ca3af;font-size:12px;">
        دليلك AI — مساعدك القانوني والحكومي
      </p>
    </div>
    """

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _RESEND_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": sender,
                    "to": [to_email],
                    "subject": "دليلك AI — إعادة تعيين كلمة المرور",
                    "html": body_html,
                },
            )
        if resp.status_code in (200, 201):
            # Log only a prefix of the address — never the full email
            logger.info(
                "[email_service] Reset email sent (status %d)", resp.status_code
            )
            return True
        logger.warning(
            "[email_service] Resend API returned HTTP %d — email not sent",
            resp.status_code,
        )
        return False

    except Exception as exc:
        logger.warning("[email_service] Failed to send reset email: %s", exc)
        return False
