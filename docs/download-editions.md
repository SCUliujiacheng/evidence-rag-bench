# Desktop and Android editions

The downloads include the fixed English and Chinese document snapshots. You can
ask your own questions without an API key, a model download, or an internet
connection. Desktop source links open the original websites when you are online.
Android keeps navigation inside the app; external source links do not open, but
the saved original passages remain readable in the result cards.

The desktop edition starts the existing local API and opens it in your browser.
Its launch page also has a Quit button: closing the search tab does not stop the
local service. Evaluation reports go into your operating system's user data
directory, outside the installed application. The resource files can remain
read-only, and the launcher does not rely on your current directory.

The Android edition is an offline evidence viewer. It calculates BM25, TF-IDF
and reciprocal-rank fusion for each new question on the device. It uses the
same corpus chunks, fitted index parameters and development-set refusal
thresholds as the desktop reference. The exported index contains corpus
statistics and text, not a table of questions and answers. Search questions
and results are not saved.

Android does not run the full Python benchmark CLI or the optional CrossEncoder.
Its health response and answer metadata identify `portable-index-v1`; this is a
separate implementation, not a claim that scikit-learn runs inside the APK.
Full evaluation remains available in the desktop/source edition. CrossEncoder
experiments require the source installation's `semantic` extra and a model
download; they are not bundled into the regular desktop download either.

Tests compare all fixed development and test questions, plus additional unseen
English, Chinese, mixed-script, repeated-term and out-of-vocabulary questions.
They check top-10 ordering and scores for all three retrievers, and answer/refusal
behavior, evidence IDs and citations. Floating-point score tolerance is
`1e-12` relative / `1e-14` absolute. A score within that tolerance of the exported
threshold is treated as equal to it. This is a regression check over these inputs,
not proof of semantic correctness for every future question.

The APK contains the document snapshots and license notices, but not the fixed
evaluation questions. Desktop resources additionally contain the evaluation
cases required by the API. Generated packages never include the developer's
reports, untracked corpus files, virtual environment, or private settings.

Build the public resources with `uv run python distribution/prepare_android.py`.
This requires the shared Android host in `distribution/portable_host.py`; it
creates only the Android project's generated build directories. Desktop builds
use the same resource exporter. Build metadata records the source revision.

The first cross-platform packages are prereleases pending physical-device
testing and publisher certificates. Automated executable and emulator checks
are reported with each release; they do not imply signing or device checks that
have not been performed.
