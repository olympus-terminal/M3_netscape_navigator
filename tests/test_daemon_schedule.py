import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, AsyncMock, call, patch

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


class StartupInstallerTests(unittest.TestCase):
    @staticmethod
    def _make_executable(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    def test_service_preserves_lexical_venv_path_and_escapes_systemd_specifiers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_dir = root / "moved repo %i"
            project_dir.mkdir()
            (project_dir / "agent.py").write_text("# agent\n", encoding="utf-8")

            real_python = root / "runtime" / "python3"
            self._make_executable(real_python)
            venv_python = project_dir / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(real_python)

            unit = agent._render_startup_unit(project_dir, venv_python)

        escaped_project = (
            str(project_dir.absolute()).replace(" ", r"\x20").replace("%", "%%")
        )
        exec_project = str(project_dir.absolute()).replace("%", "%%")
        escaped_python = str(venv_python.absolute()).replace("%", "%%")
        self.assertIn(f"WorkingDirectory={escaped_project}", unit)
        self.assertIn(
            f'ExecStart="{escaped_python}" "{exec_project}/agent.py" --daemon',
            unit,
        )
        self.assertNotIn(str(real_python.absolute()), unit)
        self.assertIn("Restart=always", unit)
        self.assertIn("WantedBy=default.target", unit)

    def test_desktop_fallback_applies_both_string_and_exec_escaping_layers(self):
        systemctl_path = Path(
            '/opt/system ctl $cash\\slash"quote%percent/bin/systemctl'
        )

        desktop = agent._render_desktop_autostart(systemctl_path)
        exec_line = next(
            line for line in desktop.splitlines() if line.startswith("Exec=")
        )

        self.assertIn(r"\\$cash", exec_line)
        self.assertIn(r"\\\\slash", exec_line)
        self.assertIn(r'\\"quote', exec_line)
        self.assertIn("quote%%percent", exec_line)

    def test_install_writes_startup_files_and_runs_checked_argv_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_dir = root / "repo"
            project_dir.mkdir()
            (project_dir / "agent.py").write_text("# agent\n", encoding="utf-8")
            python_path = project_dir / ".venv" / "bin" / "python"
            systemctl_path = root / "bin" / "systemctl"
            systemd_analyze_path = root / "bin" / "systemd-analyze"
            self._make_executable(python_path)
            self._make_executable(systemctl_path)
            self._make_executable(systemd_analyze_path)
            config_home = root / "config"

            with patch.object(agent.subprocess, "run") as run:
                service_path, desktop_path, changed = agent.install_startup_service(
                    project_dir=project_dir,
                    python_executable=python_path,
                    config_home=config_home,
                    systemctl_path=systemctl_path,
                    systemd_analyze_path=systemd_analyze_path,
                )

            self.assertTrue(changed)
            self.assertEqual(
                service_path,
                config_home / "systemd" / "user" / "ai-research-agent.service",
            )
            self.assertEqual(
                desktop_path,
                config_home / "autostart" / "ai-research-agent.desktop",
            )
            self.assertIn(str(project_dir), service_path.read_text(encoding="utf-8"))
            self.assertIn(str(python_path), service_path.read_text(encoding="utf-8"))
            desktop = desktop_path.read_text(encoding="utf-8")
            self.assertIn("Exec=/bin/sh -c", desktop)
            self.assertIn(
                f"{systemctl_path} --user daemon-reload",
                desktop,
            )
            self.assertIn(
                f"{systemctl_path} --user start ai-research-agent.service",
                desktop,
            )
            self.assertEqual(service_path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(desktop_path.stat().st_mode & 0o777, 0o644)
            self.assertGreater(agent._SYSTEMCTL_TIMEOUT_SECONDS, 120)
            expected = [
                call(
                    [
                        str(systemd_analyze_path),
                        "--user",
                        "verify",
                        ANY,
                    ],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
                call(
                    [str(systemctl_path), "--user", "daemon-reload"],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
                call(
                    [
                        str(systemctl_path),
                        "--user",
                        "enable",
                        "ai-research-agent.service",
                    ],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
                call(
                    [
                        str(systemctl_path),
                        "--user",
                        "restart",
                        "ai-research-agent.service",
                    ],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
                call(
                    [
                        str(systemctl_path),
                        "--user",
                        "is-enabled",
                        "--quiet",
                        "ai-research-agent.service",
                    ],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
                call(
                    [
                        str(systemctl_path),
                        "--user",
                        "is-active",
                        "--quiet",
                        "ai-research-agent.service",
                    ],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
            ]
            self.assertEqual(run.call_args_list, expected)

    def test_second_install_is_idempotent_and_does_not_restart_unchanged_service(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_dir = root / "repo"
            project_dir.mkdir()
            (project_dir / "agent.py").write_text("# agent\n", encoding="utf-8")
            python_path = project_dir / ".venv" / "bin" / "python"
            systemctl_path = root / "bin" / "systemctl"
            systemd_analyze_path = root / "bin" / "systemd-analyze"
            self._make_executable(python_path)
            self._make_executable(systemctl_path)
            self._make_executable(systemd_analyze_path)
            config_home = root / "config"

            with patch.object(agent.subprocess, "run"):
                service_path, desktop_path, _ = agent.install_startup_service(
                    project_dir=project_dir,
                    python_executable=python_path,
                    config_home=config_home,
                    systemctl_path=systemctl_path,
                    systemd_analyze_path=systemd_analyze_path,
                )
            original_service = service_path.read_bytes()
            original_desktop = desktop_path.read_bytes()

            with (
                patch.object(agent.os, "replace") as replace,
                patch.object(agent.subprocess, "run") as run,
            ):
                _, _, changed = agent.install_startup_service(
                    project_dir=project_dir,
                    python_executable=python_path,
                    config_home=config_home,
                    systemctl_path=systemctl_path,
                    systemd_analyze_path=systemd_analyze_path,
                )

            self.assertFalse(changed)
            replace.assert_not_called()
            self.assertEqual(service_path.read_bytes(), original_service)
            self.assertEqual(desktop_path.read_bytes(), original_desktop)
            self.assertIn(
                call(
                    [str(systemctl_path), "--user", "daemon-reload"],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
                run.call_args_list,
            )
            self.assertIn(
                call(
                    [
                        str(systemctl_path),
                        "--user",
                        "start",
                        "ai-research-agent.service",
                    ],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
                run.call_args_list,
            )

    def test_atomic_write_failure_preserves_existing_startup_file(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "service"
            destination.write_text("old config\n", encoding="utf-8")

            with patch.object(
                agent.os,
                "replace",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    agent._write_text_if_changed_atomic(
                        destination,
                        "new config\n",
                    )

            self.assertEqual(destination.read_text(encoding="utf-8"), "old config\n")
            self.assertEqual(
                [path for path in destination.parent.iterdir() if path != destination],
                [],
            )

    def test_unchanged_group_writable_startup_file_is_republished_as_0644(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "service"
            destination.write_text("same config\n", encoding="utf-8")
            destination.chmod(0o664)

            changed = agent._write_text_if_changed_atomic(
                destination,
                "same config\n",
            )

            self.assertTrue(changed)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o644)

    def test_reinstall_after_move_replaces_old_repo_and_venv_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            systemctl_path = root / "bin" / "systemctl"
            systemd_analyze_path = root / "bin" / "systemd-analyze"
            self._make_executable(systemctl_path)
            self._make_executable(systemd_analyze_path)
            config_home = root / "config"

            projects = []
            for name in ("old repo", "new repo"):
                project_dir = root / name
                project_dir.mkdir()
                (project_dir / "agent.py").write_text("# agent\n", encoding="utf-8")
                python_path = project_dir / ".venv" / "bin" / "python"
                self._make_executable(python_path)
                projects.append((project_dir, python_path))

            with patch.object(agent.subprocess, "run"):
                service_path, _, _ = agent.install_startup_service(
                    project_dir=projects[0][0],
                    python_executable=projects[0][1],
                    config_home=config_home,
                    systemctl_path=systemctl_path,
                    systemd_analyze_path=systemd_analyze_path,
                )

            with patch.object(agent.subprocess, "run") as run:
                _, _, changed = agent.install_startup_service(
                    project_dir=projects[1][0],
                    python_executable=projects[1][1],
                    config_home=config_home,
                    systemctl_path=systemctl_path,
                    systemd_analyze_path=systemd_analyze_path,
                )

            unit = service_path.read_text(encoding="utf-8")
            self.assertTrue(changed)
            self.assertNotIn(str(projects[0][0]), unit)
            self.assertNotIn(str(projects[0][1]), unit)
            self.assertIn(str(projects[1][0]), unit)
            self.assertIn(str(projects[1][1]), unit)
            self.assertIn(
                call(
                    [
                        str(systemctl_path),
                        "--user",
                        "restart",
                        "ai-research-agent.service",
                    ],
                    check=True,
                    timeout=agent._SYSTEMCTL_TIMEOUT_SECONDS,
                ),
                run.call_args_list,
            )

    def test_missing_interpreter_fails_before_writing_startup_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_dir = root / "repo"
            project_dir.mkdir()
            (project_dir / "agent.py").write_text("# agent\n", encoding="utf-8")
            systemctl_path = root / "systemctl"
            self._make_executable(systemctl_path)
            config_home = root / "config"

            with patch.object(agent.subprocess, "run") as run:
                with self.assertRaisesRegex(FileNotFoundError, "Python interpreter"):
                    agent.install_startup_service(
                        project_dir=project_dir,
                        python_executable=project_dir / ".venv" / "bin" / "python",
                        config_home=config_home,
                        systemctl_path=systemctl_path,
                    )

            self.assertFalse(config_home.exists())
            run.assert_not_called()

    def test_relative_xdg_config_home_is_ignored(self):
        with (
            patch.dict(os.environ, {"XDG_CONFIG_HOME": "relative/config"}),
            patch.object(Path, "home", return_value=Path("/safe/home")),
        ):
            config_home = agent._startup_config_home()

        self.assertEqual(config_home, Path("/safe/home/.config"))

    def test_unit_validation_failure_preserves_existing_startup_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_dir = root / "repo"
            project_dir.mkdir()
            (project_dir / "agent.py").write_text("# agent\n", encoding="utf-8")
            python_path = project_dir / "python"
            systemctl_path = root / "systemctl"
            systemd_analyze_path = root / "systemd-analyze"
            self._make_executable(python_path)
            self._make_executable(systemctl_path)
            self._make_executable(systemd_analyze_path)
            config_home = root / "config"
            service_path = (
                config_home / "systemd" / "user" / "ai-research-agent.service"
            )
            desktop_path = (
                config_home / "autostart" / "ai-research-agent.desktop"
            )
            service_path.parent.mkdir(parents=True)
            desktop_path.parent.mkdir(parents=True)
            service_path.write_text("known-good service\n", encoding="utf-8")
            desktop_path.write_text("known-good desktop\n", encoding="utf-8")

            validation_error = subprocess.CalledProcessError(
                1,
                "systemd-analyze",
            )
            with patch.object(
                agent.subprocess,
                "run",
                side_effect=validation_error,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    agent.install_startup_service(
                        project_dir=project_dir,
                        python_executable=python_path,
                        config_home=config_home,
                        systemctl_path=systemctl_path,
                        systemd_analyze_path=systemd_analyze_path,
                    )

            self.assertEqual(
                service_path.read_text(encoding="utf-8"),
                "known-good service\n",
            )
            self.assertEqual(
                desktop_path.read_text(encoding="utf-8"),
                "known-good desktop\n",
            )

    def test_systemctl_failure_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project_dir = root / "repo"
            project_dir.mkdir()
            (project_dir / "agent.py").write_text("# agent\n", encoding="utf-8")
            python_path = project_dir / "python"
            systemctl_path = root / "systemctl"
            systemd_analyze_path = root / "systemd-analyze"
            self._make_executable(python_path)
            self._make_executable(systemctl_path)
            self._make_executable(systemd_analyze_path)

            with patch.object(
                agent.subprocess,
                "run",
                side_effect=[
                    None,
                    subprocess.CalledProcessError(1, "systemctl"),
                ],
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    agent.install_startup_service(
                        project_dir=project_dir,
                        python_executable=python_path,
                        config_home=root / "config",
                        systemctl_path=systemctl_path,
                        systemd_analyze_path=systemd_analyze_path,
                    )

    def test_install_service_cli_short_circuits_the_research_pipeline(self):
        installed = (
            Path("/config/systemd/user/ai-research-agent.service"),
            Path("/config/autostart/ai-research-agent.desktop"),
            False,
        )
        with (
            patch.object(sys, "argv", ["agent.py", "--install-service"]),
            patch.object(agent, "install_startup_service", return_value=installed) as install,
            patch.object(agent, "setup_logging") as setup_logging,
            patch.object(agent.asyncio, "run") as run_async,
            patch("builtins.print"),
        ):
            agent.main()

        install.assert_called_once_with()
        setup_logging.assert_not_called()
        run_async.assert_not_called()


if __name__ == "__main__":
    unittest.main()
