# Download and install

Choose the `v0.3.0` prerelease from [GitHub Releases](https://github.com/SCUliujiacheng/evidence-rag-bench/releases).
The release page lists the attachments produced by successful platform builds,
their smoke-test reports, and `SHA256SUMS`. The automatically generated “Source
code” archives are for development; use the named application attachment below.

| Device | Application attachment |
| --- | --- |
| Windows, 64-bit Intel/AMD | `evidence-rag-bench-0.3.0-windows-x64.zip` |
| Mac with Apple silicon | `evidence-rag-bench-0.3.0-macos-arm64.zip` |
| Mac with Intel processor | `evidence-rag-bench-0.3.0-macos-x64.zip` |
| Android 7.0+ on arm64, or an x86_64 emulator | `evidence-rag-bench-0.3.0-android.apk` |

On Windows, extract the entire ZIP and keep the executable beside its `_internal`
folder. Open `evidence-rag-bench.exe`. On a Mac, extract the ZIP and open the
included `.app`; it can be moved to Applications. No Python installation is
needed. The launch page opens in the default browser and provides Open app and
Quit controls. Closing a browser tab does not stop the local process.

These first packages are previews. Windows builds have no publisher certificate,
and macOS builds are not Developer ID signed or notarized. The operating system
may identify them as an unknown publisher or block opening. Check the repository,
release version and `SHA256SUMS` before deciding whether to allow a preview build.
Automated smoke tests are not a claim of physical-device certification.

On Android, download the APK to the device and open it through the system installer.
Android may ask you to allow installation from the app you used to download it.
It is a standalone offline application, not a Play Store listing. The package
supports 64-bit devices; it does not support older 32-bit Android phones.

Both editions show the same English/Chinese corpus selector and evidence viewer.
You can enter new questions without a key or an internet connection. The desktop
edition also exposes the evaluation API. The Android edition runs the portable
lexical retrieval engine; it does not run the full benchmark CLI or CrossEncoder.
The optional semantic reranker is only available through the source installation,
not either ordinary download. See [edition details](download-editions.md).

In Android, navigation stays within the local app: external GitHub and source
links do not open another website. The retrieved original text is still available
inside each result. Desktop source links can open upstream websites when online.

Desktop evaluation reports are separate from the installed app:

| Platform | Report directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\evidence-rag-bench\artifacts\reports` |
| macOS | `~/Library/Application Support/evidence-rag-bench/artifacts/reports` |

The Android viewer does not save questions or answers; its runtime files stay in
the app's private storage. Replacing a desktop application folder does not remove
its user-data directory. Copy reports you want to retain before deleting that
directory or uninstalling the Android app. Installation resources never serve as
the writable report directory.
