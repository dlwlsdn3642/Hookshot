"""Regression guard for the Windows Builder's local-only (None) update track.

Checks the path that must not download the official Builder. No game data or
Windows GUI is required; the WPF project is compiled separately.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "builder" / "HookshotBuilder"
UPDATER = (ROOT / "Updater.cs").read_text(encoding="utf-8")
WINDOW = (ROOT / "MainWindow.xaml.cs").read_text(encoding="utf-8")
XAML = (ROOT / "MainWindow.xaml").read_text(encoding="utf-8")
SETTINGS = (ROOT / "BuilderSettings.cs").read_text(encoding="utf-8")


def section(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


class NoneTrack(unittest.TestCase):
    def test_none_is_selectable_and_persistent(self):
        self.assertIn('None = "none"', UPDATER)
        self.assertIn('Updater.None };', WINDOW)
        self.assertIn('"None (this builder)"', WINDOW)
        self.assertIn('saved != null && TrackIds.Contains(saved)', WINDOW)
        self.assertIn('Track = chosenTrack ?? BuilderSettings.Load(SettingsPath)?.Track', WINDOW)
        self.assertIn('public string? Track { get; set; }', SETTINGS)
        self.assertIn('skip all builder updates', XAML)

    def test_none_makes_no_release_requests(self):
        latest = section(UPDATER, 'public static async Task<string?> LatestTag(string track)', 'public static async Task<string?> LatestTag()')
        self.assertLess(latest.index('if (track == None) return null;'), latest.index('new HttpClient'))
        target = section(UPDATER, 'public static string? Target(', 'public static async Task<string?> LatestTag(string track)')
        self.assertIn('if (track == None) return tools;', target)
        self.assertIn('if (LocalOnly) mode = game != null && !sourcePicked ? Mode.RebuildLocal : Mode.Build;', WINDOW)
        self.assertIn('var tag = track == Updater.None ? null :', WINDOW)

    def test_existing_game_rebuilds_using_current_tools(self):
        buttons = section(WINDOW, 'void RefreshMode()', 'static string? ReadVersion(')
        self.assertIn('case Mode.RebuildLocal:', buttons)
        self.assertIn('BuildButton.Content = "Rebuild with this builder";', buttons)
        click = section(WINDOW, 'async void StartBuild(', 'const string DlssLicense')
        self.assertIn('case Mode.RebuildLocal:', click)
        self.assertIn('Updater.UpdateFolder(this, OutPath.Text, Tools.Find()?.Version, fetched: false);', click)
        updater = section(UPDATER, 'public static async void UpdateFolder(', 'static async Task<HttpResponseMessage> GetWithRetries(')
        self.assertLess(updater.index('if (w.LocalOnly)'), updater.index('GetWithRetries(w, http, api'))
        self.assertIn('w.UpdateBuild(tools, folder);', updater[updater.index('if (w.LocalOnly)'):updater.index('// (just downloaded')])
        self.assertIn('Run(tools, ShaderStep.Args(new List<string> { "--update", folder }', WINDOW)

    def test_launcher_does_not_offer_official_update_in_none_mode(self):
        launcher = section(WINDOW, 'async void ShowLauncher(', 'static string? ReadVersion(')
        self.assertIn('if (LocalOnly)', launcher)
        guard = section(launcher, 'if (LocalOnly)', 'latestTag = Environment.GetEnvironmentVariable(')
        self.assertIn('LaunchUpdate.Visibility = Visibility.Collapsed;', guard)
        self.assertIn('return;', guard)
        self.assertIn('internal bool LocalOnly => Track == Updater.None;', WINDOW)
        self.assertIn('if (Track != track) return;', WINDOW)
        self.assertIn('if (w.LocalOnly)', UPDATER)  # --update from the game follows the same guard


if __name__ == "__main__":
    unittest.main()
