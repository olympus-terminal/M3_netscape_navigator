import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import agent


GST = timezone(timedelta(hours=4))


class DaemonScheduleTests(unittest.TestCase):
    def test_legacy_last_run_defers_restart_until_daily_interval(self):
        now = datetime(2026, 7, 23, 11, 30, tzinfo=GST)
        state = {
            "last_run": "2026-07-23T11:00:00",
            "pending_digest": False,
        }

        scheduled = agent._scheduled_run_from_state(state, now)

        self.assertEqual(
            scheduled,
            datetime(2026, 7, 24, 11, 0, tzinfo=GST),
        )

    def test_pending_digest_uses_hourly_retry_interval(self):
        now = datetime(2026, 7, 23, 11, 30, tzinfo=GST)
        state = {
            "last_run": "2026-07-23T11:00:00+04:00",
            "pending_digest": True,
        }

        scheduled = agent._scheduled_run_from_state(state, now)

        self.assertEqual(
            scheduled,
            datetime(2026, 7, 23, 12, 0, tzinfo=GST),
        )

    def test_new_last_run_wins_over_stale_persisted_schedule(self):
        now = datetime(2026, 7, 23, 11, 30, tzinfo=GST)
        state = {
            "last_run": "2026-07-23T11:29:00+04:00",
            "next_run_at": "2026-07-23T11:00:00+04:00",
            "pending_digest": False,
        }

        scheduled = agent._scheduled_run_from_state(state, now)

        self.assertEqual(
            scheduled,
            datetime(2026, 7, 24, 11, 29, tzinfo=GST),
        )

    def test_failed_cycle_persists_aware_hourly_retry_without_losing_state(self):
        now = datetime(2026, 7, 23, 11, 30, tzinfo=GST)
        state = {"pending_digest": True, "seen_urls": ["https://example.test/paper"]}

        with (
            patch.object(agent, "load_state", return_value=state),
            patch.object(agent, "save_state") as save_state,
        ):
            scheduled = agent._persist_next_run(False, now)

        self.assertEqual(scheduled, datetime(2026, 7, 23, 12, 30, tzinfo=GST))
        persisted = save_state.call_args.args[0]
        self.assertEqual(persisted["pending_digest"], True)
        self.assertEqual(persisted["seen_urls"], state["seen_urls"])
        self.assertEqual(
            persisted["next_run_at"],
            "2026-07-23T12:30:00+04:00",
        )
        self.assertIsNotNone(datetime.fromisoformat(persisted["next_run_at"]).tzinfo)

    def test_successful_cycle_persists_daily_schedule(self):
        now = datetime(2026, 7, 23, 11, 30, tzinfo=GST)

        with (
            patch.object(agent, "load_state", return_value={}),
            patch.object(agent, "save_state") as save_state,
        ):
            scheduled = agent._persist_next_run(True, now)

        self.assertEqual(scheduled, datetime(2026, 7, 24, 11, 30, tzinfo=GST))
        self.assertEqual(
            save_state.call_args.args[0]["next_run_at"],
            "2026-07-24T11:30:00+04:00",
        )


class DaemonRestartTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_waits_instead_of_immediately_running_cycle(self):
        now = datetime(2026, 7, 23, 11, 30, tzinfo=GST)
        state = {
            "last_run": "2026-07-23T11:29:00+04:00",
            "pending_digest": False,
        }
        run_cycle = AsyncMock()

        with (
            patch.object(agent, "_local_now", return_value=now),
            patch.object(agent, "load_state", return_value=state),
            patch.object(agent, "run_cycle", run_cycle),
            patch.object(agent.signal, "signal"),
        ):
            daemon_task = asyncio.create_task(agent.daemon_loop())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            run_cycle.assert_not_awaited()
            daemon_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await daemon_task

    async def test_sigterm_wakes_a_scheduled_daemon_without_running_a_cycle(self):
        now = datetime(2026, 7, 23, 11, 30, tzinfo=GST)
        state = {
            "last_run": "2026-07-23T11:29:00+04:00",
            "pending_digest": False,
        }
        handlers = {}
        run_cycle = AsyncMock()

        def capture_handler(sig, handler):
            handlers[sig] = handler

        with (
            patch.object(agent, "_local_now", return_value=now),
            patch.object(agent, "load_state", return_value=state),
            patch.object(agent, "run_cycle", run_cycle),
            patch.object(agent.signal, "signal", side_effect=capture_handler),
        ):
            daemon_task = asyncio.create_task(agent.daemon_loop())
            await asyncio.sleep(0)
            handlers[agent.signal.SIGTERM](agent.signal.SIGTERM, None)
            await asyncio.wait_for(daemon_task, timeout=1)

        run_cycle.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
