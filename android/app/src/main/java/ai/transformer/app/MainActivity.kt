package ai.transformer.app

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity

/**
 * Transformer Android shell.
 *
 * Loads the live web app (GitHub Pages). The 1M-parameter model itself runs
 * in the page via onnxruntime-web; chat history is stored in the WebView's
 * DOM storage — it stays on the device, exactly like in the browser.
 *
 * For a fully offline build: copy the contents of `web/` into
 * `app/src/main/assets/site/` and load "file:///android_asset/site/index.html".
 */
class MainActivity : AppCompatActivity() {

    companion object {
        const val APP_URL = "https://samratbarman1013-commits.github.io/transformer-llm/"
    }

    private lateinit var web: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        web = WebView(this)
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true          // on-device chat history
            allowFileAccess = false
            allowContentAccess = false
        }
        web.webViewClient = WebViewClient()
        setContentView(web)
        web.loadUrl(APP_URL)
    }

    override fun onBackPressed() {
        if (web.canGoBack()) web.goBack() else super.onBackPressed()
    }
}
