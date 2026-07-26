import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
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

    def test_successful_cycle_uses_start_time_instead_of_completion_time(self):
        completed_at = datetime(2026, 7, 23, 11, 30, tzinfo=GST)
        state = {
            "last_run": "2026-07-23T10:00:00+04:00",
            "seen_urls": ["https://example.test/paper"],
        }

        with (
            patch.object(agent, "load_state", return_value=state),
            patch.object(agent, "save_state") as save_state,
        ):
            scheduled = agent._persist_next_run(True, completed_at)

        self.assertEqual(scheduled, datetime(2026, 7, 24, 10, 0, tzinfo=GST))
        self.assertEqual(
            save_state.call_args.args[0]["next_run_at"],
            "2026-07-24T10:00:00+04:00",
        )
        self.assertEqual(save_state.call_args.args[0]["seen_urls"], state["seen_urls"])

    def test_overlong_success_rolls_forward_to_first_future_daily_slot(self):
        completed_at = datetime(2026, 7, 25, 10, 0, tzinfo=GST)
        state = {"last_run": "2026-07-23T10:00:00+04:00"}

        with (
            patch.object(agent, "load_state", return_value=state),
            patch.object(agent, "save_state") as save_state,
        ):
            scheduled = agent._persist_next_run(True, completed_at)

        self.assertEqual(scheduled, datetime(2026, 7, 26, 10, 0, tzinfo=GST))
        self.assertEqual(
            save_state.call_args.args[0]["next_run_at"],
            "2026-07-26T10:00:00+04:00",
        )


class DaemonRestartTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_cycle_records_its_start_as_schedule_anchor(self):
        started_at = datetime(2026, 7, 23, 10, 0, tzinfo=GST)

        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(agent, "_local_now", return_value=started_at),
                patch.object(agent, "preflight_codex", AsyncMock()),
                patch.object(agent, "load_state", return_value={"seen_urls": [], "runs": 0}),
                patch.object(agent, "save_state") as save_state,
                patch.object(agent, "judge_loop", AsyncMock()),
                patch.object(
                    agent,
                    "_run_crawlers_with_judge_supervision",
                    AsyncMock(),
                ),
                patch.object(agent, "_drain_judges", AsyncMock()),
                patch.object(agent, "RAW_DIR", Path(temp)),
            ):
                cycle_success = await agent.run_cycle(crawl=True, digest=False)

        self.assertTrue(cycle_success)
        self.assertEqual(
            save_state.call_args.args[0]["last_run"],
            "2026-07-23T10:00:00+04:00",
        )

    async def test_wall_clock_jump_past_deadline_ends_chunked_wait(self):
        before_suspend = datetime(2026, 7, 23, 11, 0, tzinfo=GST)
        after_resume = datetime(2026, 7, 23, 13, 0, tzinfo=GST)
        target = datetime(2026, 7, 23, 12, 0, tzinfo=GST)
        shutdown = asyncio.Event()
        observed_timeouts = []

        async def monotonic_timeout(awaitable, timeout):
            awaitable.close()
            observed_timeouts.append(timeout)
            raise asyncio.TimeoutError

        with (
            patch.object(
                agent,
                "_local_now",
                side_effect=[before_suspend, after_resume],
            ),
            patch.object(agent.asyncio, "wait_for", new=monotonic_timeout),
        ):
            await agent._wait_until(target, shutdown)

        self.assertEqual(observed_timeouts, [agent._SCHEDULE_RECHECK_SECONDS])

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
