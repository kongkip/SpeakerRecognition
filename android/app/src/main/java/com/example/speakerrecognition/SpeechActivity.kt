package com.example.speakerrecognition

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.*
import android.util.Log
import android.view.View
import android.view.ViewTreeObserver
import android.widget.CompoundButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.content.ContextCompat
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetBehavior.BottomSheetCallback
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.locks.ReentrantLock
import kotlin.math.roundToInt

class SpeechActivity : AppCompatActivity(), CompoundButton.OnCheckedChangeListener,
        View.OnClickListener {

    val TAG = "SpeechActivity"
    // Constants that control the behavior of the recognition code and model
    // settings. See the audio recognition tutorial for a detailed explanation of
    // all these, but you should customize them to match your training settings if
    // you are running your own model.
    val SAMPLE_RATE = 16000
    val SAMPLE_DURATION_MS = 1000
    val RECORDING_LENGTH = (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000)
    val AVERAGE_WINDOW_DURATION_MS: Long = 1000
    val DETECTION_THRESHOLD = 0.50f
    val SUPPRESSION_MS = 1500
    val MINIMUM_COUNT = 3
    private val MINIMUM_TIME_BETWEEN_SAMPLES_MS: Long = 30
    private val LABEL_FILENAME = "file:///android_asset/conv_labels.txt"
    private val MODEL_FILENAME = "file:///android_asset/tflite_model.tflite"

    // UI elements.
    private val REQUEST_RECORD_AUDIO = 13
    private val LOG_TAG = SpeechActivity::class.java.simpleName

    // Working variables.
    var recordingBuffer = ShortArray(RECORDING_LENGTH)
    var recordingOffset = 0
    var shouldContinue = true
    private var recordingThread: Thread? = null
    var shouldContinueRecognition = true
    private var recognitionThread: Thread? = null
    private val recordingBufferLock = ReentrantLock()

    private var labels: List<String> = ArrayList()
    private var recognizeCommands: RecognizeCommands? = null
    private var bottomSheetLayout: LinearLayout? = null
    private var gestureLayout: LinearLayout? = null
    private var sheetBehavior: BottomSheetBehavior<*>? = null

    private var tfLite: Interpreter? = null
    private var bottomSheetArrowImageView: ImageView? = null

    private var benjaminTextView: TextView? = null
    private var jenTextView: TextView? = null
    private var juliaTextView: TextView? = null
    private var margaretTextView: TextView? = null
    private var nelsonTextView: TextView? = null
    private var sampleRateTextView: TextView? = null
    private  var inferenceTimeTextView: TextView? = null
    private var plusImageView: ImageView? = null
    private  var minusImageView:ImageView? = null
    private var apiSwitchCompat: SwitchCompat? = null
    private var threadsTextView: TextView? = null
    private var lastProcessingTimeMs: Long = 0
    private val handler = Handler()
    private var selectedTextView: TextView? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_speech)


        val actualLabelFilename = LABEL_FILENAME.split("file:///android_asset/").
                dropLastWhile { it.isEmpty() }.toTypedArray()[1]
        Log.i(LOG_TAG, "Reading labels from:$actualLabelFilename")
        labels = loadModelLabels(actualLabelFilename,assets)
        recognizeCommands = RecognizeCommands(
                labels, AVERAGE_WINDOW_DURATION_MS, DETECTION_THRESHOLD,
                SUPPRESSION_MS, MINIMUM_COUNT, MINIMUM_TIME_BETWEEN_SAMPLES_MS)

        val actualModelFilename = MODEL_FILENAME.split(
                "file:///android_asset/").dropLastWhile { it.isEmpty() }.toTypedArray()[1]
        val options = Interpreter.Options()
        tfLite = Interpreter(loadModelFile(actualModelFilename, assets), options)
        tfLite!!.resizeInput(0, intArrayOf(1, 1, RECORDING_LENGTH))

        requestMicrophonePermission()
        startRecording()
        startRecognition()

        sampleRateTextView = findViewById(R.id.sample_rate)
        inferenceTimeTextView = findViewById(R.id.inference_info)
        bottomSheetLayout = findViewById(R.id.bottom_sheet_layout)
        gestureLayout = findViewById(R.id.gesture_layout)
        sheetBehavior = BottomSheetBehavior.from<LinearLayout>(bottomSheetLayout)
        bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow)

        threadsTextView = findViewById(R.id.threads)
        plusImageView = findViewById(R.id.plus)
        minusImageView = findViewById(R.id.minus)
        apiSwitchCompat = findViewById(R.id.api_info_switch)

        benjaminTextView = findViewById(R.id.benjamin_netanyau)
        jenTextView = findViewById(R.id.jens_stoltenberg)
        juliaTextView = findViewById(R.id.julia_gillard)
        margaretTextView = findViewById(R.id.magaret_tarcher)
        nelsonTextView = findViewById(R.id.nelson_mandela)

        apiSwitchCompat!!.setOnCheckedChangeListener(this)

        val vto : ViewTreeObserver = gestureLayout!!.viewTreeObserver
        vto.addOnGlobalLayoutListener{
            gestureLayout!!.viewTreeObserver.removeOnGlobalLayoutListener{ }
            val height: Int = gestureLayout!!.height
            this.sheetBehavior!!.peekHeight = height
        }
        sheetBehavior!!.isHideable = false

        sheetBehavior!!.setBottomSheetCallback(
                object : BottomSheetCallback() {
                    @SuppressLint("SwitchIntDef")
                    override fun onStateChanged(bottomSheet: View, newState: Int) {
                        when (newState) {
                            BottomSheetBehavior.STATE_HIDDEN -> {
                            }
                            BottomSheetBehavior.STATE_EXPANDED -> {
                                bottomSheetArrowImageView!!.setImageResource(R.drawable.icn_chevron_down)
                            }
                            BottomSheetBehavior.STATE_COLLAPSED -> {
                                bottomSheetArrowImageView!!.setImageResource(R.drawable.icn_chevron_up)
                            }
                            BottomSheetBehavior.STATE_DRAGGING -> {
                            }
                            BottomSheetBehavior.STATE_SETTLING ->
                                bottomSheetArrowImageView!!.setImageResource(R.drawable.icn_chevron_up)
                        }
                    }

                    override fun onSlide(bottomSheet: View, slideOffset: Float) {}
                })
        plusImageView!!.setOnClickListener(this)
        minusImageView!!.setOnClickListener(this)

        sampleRateTextView!!.text = SAMPLE_RATE.toString() + " Hz"
    }

    private fun loadModelLabels(labelFileName:String, assets: AssetManager): List<String> {
        val labels: MutableList<String> = mutableListOf()
        try {
            val br = BufferedReader(InputStreamReader(assets.open(labelFileName)))
            br.useLines { lines -> lines.forEach { line -> labels.add(line) } }
        } catch (e: IOException) {
            throw RuntimeException("Problem reading label file!", e)
        }
        return labels
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelFilename: String, assets: AssetManager): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }




    private fun requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>,
                                            grantResults: IntArray) {
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.isNotEmpty() && grantResults[0] ==
                PackageManager.PERMISSION_GRANTED) {
            startRecording()
            startRecognition()
        }
    }

    @Synchronized
    fun startRecording() {
        if (recordingThread != null) {
            return
        }
        shouldContinue = true
        recordingThread = Thread(
                Runnable { record() })
        recordingThread!!.start()
    }

    @Synchronized
    fun stopRecording() {
        if (recordingThread == null) {
            return
        }
        shouldContinue = false
        recordingThread = null
    }

    private fun record() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
        // Estimate the buffer size we'll need for this device.
        var bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2
        }
        val audioBuffer = ShortArray(bufferSize / 2)
        val record = AudioRecord(
                MediaRecorder.AudioSource.DEFAULT,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize)
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!")
            return
        }
        record.startRecording()
        Log.v(LOG_TAG, "Start recording")
        // Loop, gathering audio data and copying it to a round-robin buffer.
        while (shouldContinue) {
            val numberRead = record.read(audioBuffer, 0, audioBuffer.size)
            val maxLength = recordingBuffer.size
            val newRecordingOffset = recordingOffset + numberRead
            val secondCopyLength = 0.coerceAtLeast(newRecordingOffset - maxLength)
            val firstCopyLength = numberRead - secondCopyLength
            // We store off all the data for the recognition thread to access. The ML
            // thread will copy out of this buffer into its own, while holding the
            // lock, so this should be thread safe.
            recordingBufferLock.lock()
            recordingOffset = try {
                System.arraycopy(audioBuffer, 0, recordingBuffer,
                        recordingOffset, firstCopyLength)
                System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer,
                        0, secondCopyLength)
                newRecordingOffset % maxLength
            } finally {
                recordingBufferLock.unlock()
            }
        }
        record.stop()
        record.release()
    }

    @Synchronized
    fun startRecognition() {
        if (recognitionThread != null) {
            return
        }
        shouldContinueRecognition = true
        recognitionThread = Thread(
                Runnable { recognize() })
        recognitionThread!!.start()
    }

    @Synchronized
    fun stopRecognition() {
        if (recognitionThread == null) {
            return
        }
        shouldContinueRecognition = false
        recognitionThread = null
    }

    private fun recognize() {
        Log.v(LOG_TAG, "Start recognition")
        val inputBuffer = ShortArray(RECORDING_LENGTH)
        val floatInputBuffer = Array(RECORDING_LENGTH) { FloatArray(1) }
        val outputScores = Array(1) { FloatArray(labels.size) }

        // Loop, grabbing recorded data and running the recognition model on it.
        while (shouldContinueRecognition) {
            val startTime = Date().time
            // The recording thread places data in this round-robin buffer, so lock to
            // make sure there's no writing happening and then copy it to our own
            // local version.
            recordingBufferLock.lock()
            try {
                val maxLength = recordingBuffer.size
                val firstCopyLength = maxLength - recordingOffset
                val secondCopyLength = recordingOffset
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer,
                        0, firstCopyLength)
                System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength,
                        secondCopyLength)
            } finally {
                recordingBufferLock.unlock()
            }
            // We need to feed in float values between -1.0f and 1.0f, so divide the
            // signed 16-bit inputs.
            for (i in 0 until RECORDING_LENGTH) {
                floatInputBuffer[i][0] = inputBuffer[i] / 32767.0f
            }
            val inputArray = arrayOf<Any>(floatInputBuffer)
            val outputMap: MutableMap<Int, Any> = HashMap()
            outputMap[0] = outputScores

            // Run the model.
            tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)

            // Use the smoother to figure out if we've had a real recognition event.
            val currentTime = System.currentTimeMillis()
            val result = recognizeCommands!!.processLatestResults(
                    outputScores[0], currentTime)
            lastProcessingTimeMs = Date().time - startTime
            runOnUiThread {
                inferenceTimeTextView!!.text = "$lastProcessingTimeMs ms"
                // If we do have a new command, highlight the right list entry.
                if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
                    var labelIndex = -1
                    for (i in labels.indices) {
                        if (labels[i] == result.foundCommand) {
                            labelIndex = i
                        }
                    }
                    when (labelIndex - 2) {
                        0 -> selectedTextView = benjaminTextView
                        1 -> selectedTextView = jenTextView
                        2 -> selectedTextView = juliaTextView
                        3 -> selectedTextView = margaretTextView
                        4 -> selectedTextView = nelsonTextView
                    }
                    if (selectedTextView != null) {
                        selectedTextView!!.setBackgroundResource(
                                R.drawable.round_corner_text_bg_selected)
                        val score = (result.score * 100).roundToInt().toString() + "%"
                        selectedTextView!!.text = selectedTextView!!.text.toString() + "\n" + score
                        selectedTextView!!.setTextColor(
                                ContextCompat.getColor(this,
                                        android.R.color.holo_orange_light))
                        handler.postDelayed(
                                {
                                    val originalString = selectedTextView!!.text.
                                            toString().replace(score, "").
                                            trim { it <= ' ' }
                                    selectedTextView!!.text = originalString
                                    selectedTextView!!.setBackgroundResource(
                                            R.drawable.round_corner_text_bg_unselected)
                                    selectedTextView!!.setTextColor(
                                            ContextCompat.getColor(this,
                                                    android.R.color.darker_gray))
                                },
                                750)
                    }
                }
            }
            try { // We don't need to run too frequently, so snooze for a bit.
                Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS)
            } catch (e: InterruptedException) { // Ignore
            }
        }
        Log.v(LOG_TAG, "End recognition")
    }


    override fun onClick(v: View) {
        if (v.id == R.id.plus) {
            val threads = threadsTextView!!.text.toString().trim { it <= ' ' }
            var numThreads = threads.toInt()
            numThreads++
            threadsTextView!!.text = numThreads.toString()
            //            tfLite.setNumThreads(numThreads);
            val finalNumThreads = numThreads
            //            backgroundHandler.post(() -> tfLite.setNumThreads(finalNumThreads));
        } else if (v.id == R.id.minus) {
            val threads = threadsTextView!!.text.toString().trim { it <= ' ' }
            var numThreads = threads.toInt()
            if (numThreads == 1) {
                return
            }
            numThreads--
            threadsTextView!!.text = numThreads.toString()
            tfLite!!.setNumThreads(numThreads)
            val finalNumThreads = numThreads
            //            backgroundHandler.post(() -> tfLite.setNumThreads(finalNumThreads));
        }
    }

    override fun onCheckedChanged(buttonView: CompoundButton?, isChecked: Boolean) {
        if (isChecked) apiSwitchCompat!!.text = "NNAPI" else apiSwitchCompat!!.text = "TFLITE"
    }

    private val HANDLE_THREAD_NAME = "CameraBackground"


    private fun startBackgroundThread() {
        backgroundThread = HandlerThread(HANDLE_THREAD_NAME)
        backgroundThread!!.start()
        backgroundHandler = Handler(backgroundThread!!.getLooper())
    }

    private fun stopBackgroundThread() {
        backgroundThread!!.quitSafely()
        try {
            backgroundThread!!.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e("amlan", "Interrupted when stopping background thread", e)
        }
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()
    }

    override fun onStop() {
        super.onStop()
        stopBackgroundThread()
    }
}