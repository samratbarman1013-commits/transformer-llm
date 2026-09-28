package ai.transformer.app

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity

/**
 * Transformer Android shell — thin client.
 *
 * Loads the API app from GitHub Pages. No model is bundled or downloaded:
 * all inference happens on the owner's private API server (online-only).
 * Chat history lives in the WebView's DOM storage — it stays on the device.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        // Thin client: loads the API app (no model on device, online-only).
        const val APP_URL = "https://samratbarman1013-commits.github.io/transformer-llm/app/"
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
