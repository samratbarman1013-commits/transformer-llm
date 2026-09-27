# Transformer — Android app

A thin WebView shell around the Transformer web app. The model (ONNX, 4.3 MB)
runs **inside the WebView** via onnxruntime-web — no backend server, no API
keys, and chat history lives only in the device's DOM storage.

## Build the APK

Easiest — GitHub builds it for you:

1. Push this repo to GitHub (already configured).
2. Open the **Actions** tab → **Build Android APK** → **Run workflow**.
3. Download `transformer-apk` from the run's Artifacts — installable
   (debug-signed) `app-debug.apk`.

Locally:

```bash
# requires JDK 17 + Android SDK; or: npx @bubblewrap/cli for a TWA
cd android
gradle wrapper --gradle-version 8.7   # one-time, creates gradlew
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-debug.apk
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Fully offline variant

Copy `web/` (including `model/model.onnx`) to
`android/app/src/main/assets/site/` and change `APP_URL` in `MainActivity.kt`
to `file:///android_asset/site/index.html`. The app then needs **zero**
permissions (remove INTERNET from the manifest).

## Play Store

For a store release: add a signing config (`keytool -genkey` +
`signingConfigs.release` in `app/build.gradle`) and build
`assembleRelease`. Consider `bubblewrap` to wrap the installed PWA instead —
same result, less code.
