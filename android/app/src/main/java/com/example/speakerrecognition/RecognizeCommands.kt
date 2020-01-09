package com.example.speakerrecognition

import android.util.Log
import android.util.Pair
import java.util.*

/** Reads in results from an instantaneous audio recognition model and smoothes them over time.  */
class RecognizeCommands(
        inLabels: List<String>,
        inAverageWindowDurationMs: Long,
        inDetectionThreshold: Float,
        inSuppressionMS: Int,
        inMinimumCount: Int,
        inMinimumTimeBetweenSamplesMS: Long) {
    // Configuration settings.
    private var labels: List<String> = ArrayList()
    private val averageWindowDurationMs: Long
    private val detectionThreshold: Float
    private val suppressionMs: Int
    private val minimumCount: Int
    private val minimumTimeBetweenSamplesMs: Long
    // Working variables.
    private val previousResults: Deque<Pair<Long, FloatArray>> = ArrayDeque()
    private var previousTopLabel: String
    private val labelsCount: Int
    private var previousTopLabelTime: Long
    private var previousTopLabelScore: Float

    /** Holds information about what's been recognized.  */
    class RecognitionResult(val foundCommand: String, val score: Float, val isNewCommand: Boolean)

    private class ScoreForSorting(val score: Float, val index: Int) : Comparable<ScoreForSorting> {
        override fun compareTo(other: ScoreForSorting): Int {
            return if (score > other.score) {
                -1
            } else if (score < other.score) {
                1
            } else {
                0
            }
        }

    }

    fun processLatestResults(currentResults: FloatArray, currentTimeMS: Long): RecognitionResult {
        if (currentResults.size != labelsCount) {
            throw RuntimeException(
                    "The results for recognition should contain "
                            + labelsCount
                            + " elements, but there are "
                            + currentResults.size)
        }
        if (!previousResults.isEmpty() && currentTimeMS < previousResults.first.first) {
            throw RuntimeException(
                    "You must feed results in increasing time order, but received a timestamp of "
                            + currentTimeMS
                            + " that was earlier than the previous one of "
                            + previousResults.first.first)
        }
        var howManyResults = previousResults.size
        // Ignore any results that are coming in too frequently.
        if (howManyResults > 1) {
            val timeSinceMostRecent = currentTimeMS - previousResults.last.first
            if (timeSinceMostRecent < minimumTimeBetweenSamplesMs) {
                return RecognitionResult(previousTopLabel, previousTopLabelScore, false)
            }
        }
        // Add the latest results to the head of the queue.
        previousResults.addLast(Pair(currentTimeMS, currentResults))
        // Prune any earlier results that are too old for the averaging window.
        val timeLimit = currentTimeMS - averageWindowDurationMs
        while (previousResults.first.first < timeLimit) {
            previousResults.removeFirst()
        }
        howManyResults = previousResults.size
        // If there are too few results, assume the result will be unreliable and
// bail.
        val earliestTime = previousResults.first.first
        val samplesDuration = currentTimeMS - earliestTime
        Log.v("Number of Results: ", howManyResults.toString())
        Log.v(
                "Duration < WD/FRAC?", (samplesDuration < averageWindowDurationMs / MINIMUM_TIME_FRACTION).toString())
        if (howManyResults < minimumCount //        || (samplesDuration < (averageWindowDurationMs / MINIMUM_TIME_FRACTION))
        ) {
            Log.v("RecognizeResult", "Too few results")
            return RecognitionResult(previousTopLabel, 0.0f, false)
        }
        // Calculate the average score across all the results in the window.
        val averageScores = FloatArray(labelsCount)
        for (previousResult in previousResults) {
            val scoresTensor = previousResult.second
            var i = 0
            while (i < scoresTensor.size) {
                averageScores[i] += scoresTensor[i] / howManyResults
                ++i
            }
        }
        // Sort the averaged results in descending score order.
        val sortedAverageScores = arrayOfNulls<ScoreForSorting>(labelsCount)
        for (i in 0 until labelsCount) {
            sortedAverageScores[i] = ScoreForSorting(averageScores[i], i)
        }
        Arrays.sort(sortedAverageScores)
        // See if the latest top score is enough to trigger a detection.
        val currentTopIndex = sortedAverageScores[0]!!.index
        val currentTopLabel = labels[currentTopIndex]
        val currentTopScore = sortedAverageScores[0]!!.score
        // If we've recently had another label trigger, assume one that occurs too
// soon afterwards is a bad result.
        val timeSinceLastTop: Long = if (previousTopLabel == SILENCE_LABEL || previousTopLabelTime
                == Long.MIN_VALUE) {
            Long.MAX_VALUE
        } else {
            currentTimeMS - previousTopLabelTime
        }
        val isNewCommand: Boolean
        if (currentTopScore > detectionThreshold && timeSinceLastTop > suppressionMs) {
            previousTopLabel = currentTopLabel
            previousTopLabelTime = currentTimeMS
            previousTopLabelScore = currentTopScore
            isNewCommand = true
        } else {
            isNewCommand = false
        }
        return RecognitionResult(currentTopLabel, currentTopScore, isNewCommand)
    }

    companion object {
        private const val SILENCE_LABEL = "_silence_"
        private const val MINIMUM_TIME_FRACTION: Long = 4
    }

    init {
        labels = inLabels
        averageWindowDurationMs = inAverageWindowDurationMs
        detectionThreshold = inDetectionThreshold
        suppressionMs = inSuppressionMS
        minimumCount = inMinimumCount
        labelsCount = inLabels.size
        previousTopLabel = SILENCE_LABEL
        previousTopLabelTime = Long.MIN_VALUE
        previousTopLabelScore = 0.0f
        minimumTimeBetweenSamplesMs = inMinimumTimeBetweenSamplesMS
    }
}