package com.vkr.mlp;

import com.vkr.utils.DataLoader;

import java.io.*;
import java.util.*;

public class MLPTrainer {
    private final MLP network;
    private final List<Double> lossHistory = new ArrayList<>();
    private final List<Double> accHistory = new ArrayList<>();
    private final List<Double> valLossHistory = new ArrayList<>();
    private final List<Double> valAccHistory = new ArrayList<>();
    private final boolean isBinary;
    private int numClasses;
    private final long baseSeed;
    private final int earlyStoppingPatience;
    private final double earlyStoppingMinDelta;
    private final int inputSize;

    public MLPTrainer(int[] layerSizes, String activationFunc, double learningRate,
                      double momentum, String optimizer, long seed,
                      int earlyStoppingPatience, double earlyStoppingMinDelta) {
        this.baseSeed = seed;
        this.inputSize = layerSizes[0];
        this.network = new MLP(layerSizes, learningRate, momentum, activationFunc, optimizer, seed);
        this.numClasses = layerSizes[layerSizes.length - 1];
        this.isBinary = (numClasses == 1);
        if (isBinary) this.numClasses = 2;

        this.earlyStoppingPatience = earlyStoppingPatience;
        this.earlyStoppingMinDelta = earlyStoppingMinDelta;
    }

    public void train(DataLoader.SplitData data, int epochs, int batchSize) {
        List<double[]> trainX = new ArrayList<>(data.trainX);
        List<double[]> trainY = new ArrayList<>(data.trainY);
        List<double[]> valX = new ArrayList<>(data.valX);
        List<double[]> valY = new ArrayList<>(data.valY);

        double bestValLoss = Double.MAX_VALUE;
        int patienceCounter = 0;

        System.out.println("\nMLP Training started...");
        System.out.println("Train samples: " + trainX.size());
        System.out.println("Val samples: " + valX.size());
        System.out.println("Early stopping patience: " + earlyStoppingPatience);
        System.out.println("Early stopping minDelta: " + earlyStoppingMinDelta);

        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(trainX, trainY, epoch);

            double totalLoss = 0.0;
            int correct = 0;
            int processed = 0;

            for (int start = 0; start < trainX.size(); start += batchSize) {
                int end = Math.min(start + batchSize, trainX.size());
                int curBatchSize = end - start;

                network.clearBatchGradients();
                double batchLoss = 0.0;

                for (int k = start; k < end; k++) {
                    double[] x = trainX.get(k);
                    double[] y = trainY.get(k);

                    double[] output = network.processTrainingExample(x, y);

                    if (isBinary) {
                        double p = output[0];
                        double target = y[0];
                        batchLoss += -(target * Math.log(Math.max(p, 1e-15))
                                + (1 - target) * Math.log(Math.max(1 - p, 1e-15)));
                    } else {
                        for (int j = 0; j < output.length; j++) {
                            batchLoss += -y[j] * Math.log(Math.max(output[j], 1e-15));
                        }
                    }

                    if (getPredictedClass(output) == getActualClass(y)) {
                        correct++;
                    }
                    processed++;
                }

                network.updateParametersAfterBatch(curBatchSize);
                totalLoss += batchLoss;
            }

            double avgLoss = totalLoss / processed;
            double acc = (double) correct / processed;

            lossHistory.add(avgLoss);
            accHistory.add(acc);

            System.out.printf(
                    "Epoch %3d | Loss: %.4f | Train Acc: %.5f | Batches: %d",
                    epoch + 1, avgLoss, acc, (trainX.size() + batchSize - 1) / batchSize
            );

            double valLoss = 0.0;
            int valCorrect = 0;
            int valProcessed = 0;

            for (int i = 0; i < valX.size(); i++) {
                double[] out = network.getPrediction(valX.get(i));
                double[] trueY = valY.get(i);

                if (isBinary) {
                    double p = out[0];
                    double target = trueY[0];
                    valLoss += -(target * Math.log(Math.max(p, 1e-15))
                            + (1 - target) * Math.log(Math.max(1 - p, 1e-15)));
                } else {
                    for (int j = 0; j < out.length; j++) {
                        valLoss += -trueY[j] * Math.log(Math.max(out[j], 1e-15));
                    }
                }

                if (getPredictedClass(out) == getActualClass(trueY)) {
                    valCorrect++;
                }
                valProcessed++;
            }

            valLoss /= valProcessed;
            double valAcc = (double) valCorrect / valProcessed;

            valLossHistory.add(valLoss);
            valAccHistory.add(valAcc);

            boolean improved = (bestValLoss - valLoss) > earlyStoppingMinDelta;

            System.out.printf(" | Val Loss: %.4f | Val Acc: %.5f", valLoss, valAcc);

            if (improved) {
                bestValLoss = valLoss;
                patienceCounter = 0;
                System.out.print(" | improvement");
            } else {
                patienceCounter++;
                System.out.printf(" | no improve (%d/%d)", patienceCounter, earlyStoppingPatience);
            }
            System.out.println();

            if (patienceCounter >= earlyStoppingPatience) {
                System.out.println("Early stopping на эпохе " + (epoch + 1));
                break;
            }
        }
    }

    public EvaluationResult evaluateTestCsvStreaming(String csvPath, List<double[]> labels) throws Exception {
        int[][] confusion = getConfusionMatrixCsvStreaming(csvPath, labels);
        return buildEvaluationResult(confusion);
    }

    public int[][] getConfusionMatrixCsvStreaming(String csvPath, List<double[]> labels) throws Exception {
        int[][] confusion = new int[numClasses][numClasses];

        try (BufferedReader br = new BufferedReader(new FileReader(csvPath), 1 << 20)) {
            br.readLine(); // header
            String line;
            int idx = 0;

            while ((line = br.readLine()) != null && idx < labels.size()) {
                double[] x = parseFeaturesOnlyLine(line);
                if (x == null) {
                    continue;
                }

                if (x.length != inputSize) {
                    System.out.println("Skip row: expected " + inputSize + " features, got " + x.length);
                    continue;
                }

                double[] out = network.getPrediction(x);
                int predicted = getPredictedClass(out);
                int actual = getActualClass(labels.get(idx));

                confusion[actual][predicted]++;
                idx++;
            }
        }

        return confusion;
    }

    public EvaluationResult evaluateTestCsvStreaming(String featuresCsvPath, String originalLabeledCsvPath) throws Exception {
        int[][] confusion = getConfusionMatrixCsvStreaming(featuresCsvPath, originalLabeledCsvPath);
        return buildEvaluationResult(confusion);
    }

    public int[][] getConfusionMatrixCsvStreaming(String featuresCsvPath, String originalLabeledCsvPath) throws Exception {
        int[][] confusion = new int[numClasses][numClasses];

        try (
                BufferedReader featReader = new BufferedReader(new FileReader(featuresCsvPath), 1 << 20);
                BufferedReader labelReader = new BufferedReader(new FileReader(originalLabeledCsvPath), 1 << 20)
        ) {
            featReader.readLine();   // header features CSV
            labelReader.readLine();  // header original labeled CSV

            String featLine;
            String labelLine;

            long count = 0L;
            long skippedBadFeatures = 0L;
            long skippedBadLabels = 0L;

            while ((featLine = featReader.readLine()) != null &&
                    (labelLine = labelReader.readLine()) != null) {

                double[] x = parseFeaturesOnlyLine(featLine);
                if (x == null) {
                    skippedBadFeatures++;
                    continue;
                }

                if (x.length != inputSize) {
                    System.out.println("Skip row: expected " + inputSize + " features, got " + x.length);
                    skippedBadFeatures++;
                    continue;
                }

                int actual = parseActualClassFromOriginalCsv(labelLine);
                if (actual < 0) {
                    skippedBadLabels++;
                    continue;
                }

                double[] out = network.getPrediction(x);
                int predicted = getPredictedClass(out);

                confusion[actual][predicted]++;
                count++;

                if (count % 100_000 == 0) {
                    System.out.println("Evaluated: " + count + " rows");
                }
            }

            System.out.println("Done. Evaluated rows: " + count);
            System.out.println("Skipped rows (bad features): " + skippedBadFeatures);
            System.out.println("Skipped rows (bad labels): " + skippedBadLabels);
        }

        return confusion;
    }

    public EvaluationResult evaluateSingleCsvStreaming(String labeledCsvPath) throws Exception {
        int[][] confusion = getConfusionMatrixSingleCsvStreaming(labeledCsvPath);
        return buildEvaluationResult(confusion);
    }

    public int[][] getConfusionMatrixSingleCsvStreaming(String labeledCsvPath) throws Exception {
        int[][] confusion = new int[numClasses][numClasses];

        try (BufferedReader reader = new BufferedReader(new FileReader(labeledCsvPath), 1 << 20)) {
            String line;
            boolean first = true;
            long count = 0L;
            long skippedBadFeatures = 0L;
            long skippedBadLabels = 0L;

            while ((line = reader.readLine()) != null) {
                if (first) {
                    first = false;
                    continue;
                }

                double[] x = parseFeatureLineFromLabeledCsv(line);
                if (x == null) {
                    skippedBadFeatures++;
                    continue;
                }

                if (x.length != inputSize) {
                    System.out.println("Skip row: expected " + inputSize + " features, got " + x.length);
                    skippedBadFeatures++;
                    continue;
                }

                int actual = parseActualClassFromOriginalCsv(line);
                if (actual < 0) {
                    skippedBadLabels++;
                    continue;
                }

                double[] out = network.getPrediction(x);
                int predicted = getPredictedClass(out);

                confusion[actual][predicted]++;
                count++;

                if (count % 100_000 == 0) {
                    System.out.println("Evaluated: " + count + " rows");
                }
            }

            System.out.println("Done. Evaluated rows: " + count);
            System.out.println("Skipped rows (bad features): " + skippedBadFeatures);
            System.out.println("Skipped rows (bad labels): " + skippedBadLabels);
        }

        return confusion;
    }

    private EvaluationResult buildEvaluationResult(int[][] confusion) {
        int total = 0;
        int correct = 0;

        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                total += confusion[i][j];
                if (i == j) {
                    correct += confusion[i][j];
                }
            }
        }

        double accuracy = total == 0 ? 0.0 : (double) correct / total;

        System.out.println("\n=== Streaming Test Results ===");
        System.out.printf("Test Accuracy: %.5f (%.2f%%)\n\n", accuracy, accuracy * 100);

        String[] names;
        if (isBinary) {
            names = new String[]{"Negative", "Positive"};
        } else {
            names = new String[]{
                    "Benign",
                    "DoS",
                    "DDoS",
                    "Bot",
                    "BruteForce"
            };
        }

        System.out.println("\nClassification Report:");
        System.out.printf("%-18s %10s %10s %10s %10s\n",
                "Class", "Precision", "Recall", "F1", "Support");

        double macroP = 0.0;
        double macroR = 0.0;
        double macroF = 0.0;
        int totalSupport = 0;
        int presentClasses = 0;

        for (int i = 0; i < numClasses; i++) {
            int tp = confusion[i][i];
            int fn = 0;
            int fp = 0;

            for (int j = 0; j < numClasses; j++) {
                if (j != i) {
                    fn += confusion[i][j];
                    fp += confusion[j][i];
                }
            }

            int support = tp + fn;
            if (support == 0) {
                continue;
            }

            totalSupport += support;

            double precision = (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
            double recall = (double) tp / support;
            double f1 = (precision + recall) == 0.0
                    ? 0.0
                    : 2.0 * precision * recall / (precision + recall);

            macroP += precision;
            macroR += recall;
            macroF += f1;
            presentClasses++;

            System.out.printf(
                    "%-18s %10.4f %10.4f %10.4f %10d\n",
                    names[i], precision, recall, f1, support
            );
        }

        if (presentClasses > 0) {
            macroP /= presentClasses;
            macroR /= presentClasses;
            macroF /= presentClasses;
        } else {
            macroP = 0.0;
            macroR = 0.0;
            macroF = 0.0;
        }

        System.out.println("\nMacro Average (present classes only):");
        System.out.printf(
                "%-18s %10.4f %10.4f %10.4f %10d\n",
                "AVG", macroP, macroR, macroF, totalSupport
        );

        return new EvaluationResult(accuracy, macroP, macroR, macroF, confusion);
    }

    private double[] parseFeaturesOnlyLine(String line) {
        if (line == null || line.isEmpty()) {
            return null;
        }

        String[] parts = line.split(",");
        if (parts.length == 0) {
            return null;
        }

        double[] x = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                x[i] = Double.parseDouble(parts[i].trim());
            } catch (Exception e) {
                x[i] = 0.0;
            }
        }
        return x;
    }

    private double[] parseFeatureLineFromLabeledCsv(String line) {
        if (line == null || line.isEmpty()) {
            return null;
        }

        String[] parts = line.split(",");
        if (parts.length < 2) {
            return null;
        }

        int featureCount = parts.length - 1;
        double[] x = new double[featureCount];

        for (int i = 0; i < featureCount; i++) {
            try {
                x[i] = Double.parseDouble(parts[i].trim());
            } catch (Exception e) {
                x[i] = 0.0;
            }
        }

        return x;
    }

    private int parseActualClassFromOriginalCsv(String line) {
        if (line == null || line.isEmpty()) {
            return -1;
        }

        int lastComma = line.lastIndexOf(',');
        if (lastComma < 0 || lastComma == line.length() - 1) {
            return -1;
        }

        String labelStr = line.substring(lastComma + 1).trim();

        if (isBinary) {
            if ("1".equals(labelStr) || "true".equalsIgnoreCase(labelStr) || "positive".equalsIgnoreCase(labelStr)) {
                return 1;
            }
            return 0;
        }

        return mapClassLabel(labelStr);
    }

    private int mapClassLabel(String label) {
        switch (label) {
            case "Benign":
                return 0;
            case "DoS":
                return 1;
            case "DDoS":
                return 2;
            case "Bot":
                return 3;
            case "BruteForce":
                return 4;
            default:
                return -1;
        }
    }

    private void shuffle(List<double[]> features, List<double[]> labels, int epoch) {
        Random rnd = new Random(baseSeed + 1000 + epoch);
        for (int i = features.size() - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            Collections.swap(features, i, j);
            Collections.swap(labels, i, j);
        }
    }

    private int argmax(double[] array) {
        int idx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[idx]) idx = i;
        }
        return idx;
    }

    private int getPredictedClass(double[] output) {
        if (isBinary) {
            return (output[0] >= 0.5) ? 1 : 0;
        } else {
            return argmax(output);
        }
    }

    private int getActualClass(double[] y) {
        if (isBinary) {
            return (y[0] >= 0.5) ? 1 : 0;
        } else {
            return argmax(y);
        }
    }

    public PredictionInfo predictWithConfidence(double[] x) {
        double[] out = network.getPrediction(x);
        int predictedClass = getPredictedClass(out);
        double confidence;
        double margin;

        if (isBinary) {
            double p1 = out[0];
            double p0 = 1.0 - p1;
            confidence = (predictedClass == 1) ? p1 : p0;
            margin = Math.abs(p1 - p0);
        } else {
            confidence = out[predictedClass];
            double secondBest = getSecondLargest(out);
            margin = confidence - secondBest;
        }

        return new PredictionInfo(predictedClass, confidence, margin, out);
    }

    private double getSecondLargest(double[] arr) {
        if (arr == null || arr.length == 0) {
            return 0.0;
        }
        if (arr.length == 1) {
            return 0.0;
        }

        double first = Double.NEGATIVE_INFINITY;
        double second = Double.NEGATIVE_INFINITY;

        for (double v : arr) {
            if (v > first) {
                second = first;
                first = v;
            } else if (v > second) {
                second = v;
            }
        }

        return second;
    }

    public List<Double> getLossHistory() {
        return lossHistory;
    }

    public List<Double> getAccHistory() {
        return accHistory;
    }

    public List<Double> getValAccHistory() {
        return valAccHistory;
    }

    public List<Double> getValLossHistory() {
        return valLossHistory;
    }

    public MLP getMlp() {
        return network;
    }
}
