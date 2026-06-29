package com.vkr.som;

import com.vkr.utils.ClassLabels;
import com.vkr.utils.DataLoader;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SOMTrainer {

    private final SOM som;
    private final DataLoader.SplitData split;
    private final String distance;

    public SOMTrainer(int rows, int cols,
                      DataLoader.SplitData split,
                      String distance,
                      long seed) {
        this.split = split;
        this.distance = distance;
        this.som = new SOM(rows, cols, split.trainX, distance, seed);
    }

    public void train(int iterPhase1, int iterPhase2) {
        System.out.println("Запуск обучения SOM...");
        List<Integer> trainLabels = extractClassIndices(split.trainY);
        double mapSize = Math.max(som.getR(), som.getC());
        double lr1 = 0.5;
        double lr2 = 0.05;
        double radius1 = mapSize / 2.0;
        double radius2 = mapSize * 0.1;
        som.trainTwoPhaseBalanced(
                split.trainX,
                trainLabels,
                iterPhase1,
                iterPhase2,
                lr1,
                lr2,
                radius1,
                radius2
        );
        double[][] trainArray = split.trainX.toArray(new double[0][]);
        som.countWinnersOnDataset(trainArray);
        som.assignNeuronLabels(split.trainX, split.trainY, 0.70);
        System.out.println("Обучение SOM завершено!");
    }

    private List<Integer> extractClassIndices(List<double[]> oneHotLabels) {
        List<Integer> out = new ArrayList<>(oneHotLabels.size());
        for (double[] row : oneHotLabels) {
            int idx = 0;
            for (int i = 1; i < row.length; i++) {
                if (row[i] > row[idx]) {
                    idx = i;
                }
            }
            out.add(idx);
        }

        return out;
    }

    public List<int[]> getActiveNeuronIndices() {
        List<int[]> active = new ArrayList<>();
        for (int i = 0; i < som.getR(); i++) {
            for (int j = 0; j < som.getC(); j++) {
                if (som.getNeurons()[i][j].getWinCount() > 0) {
                    active.add(new int[]{i, j});
                }
            }
        }
        if (active.isEmpty()) {
            for (int i = 0; i < som.getR(); i++) {
                for (int j = 0; j < som.getC(); j++) {
                    active.add(new int[]{i, j});
                }
            }
        }

        return active;
    }

    public void compareQEStreaming(List<double[]> testDataset) {
        double qeTrain = som.getQe();
        double sumDist = 0.0; long count = 0L;
        for (double[] x : testDataset) {
            int[] bmu = som.findBMUCoords(x);
            SOMNeuron winner = som.getNeurons()[bmu[0]][bmu[1]];
            double dist = winner.findDist(x, distance);
            sumDist += dist; count++;
        }
        double qeTest = (count > 0) ? (sumDist / count) : Double.NaN;
        System.out.println("\n=== Оценка SOM на тестовых данных (streaming) ===");
        System.out.printf("QE на train (финал): %.4f%n", qeTrain);
        System.out.printf("QE на test: %.4f%n", qeTest);
        System.out.printf("Разница абсолютная: %.4f%n", qeTest - qeTrain);
        System.out.printf("Отношение QE_test / QE_train: %.2f (%.0f%%)%n", qeTest / qeTrain, (qeTest / qeTrain - 1) * 100.0);
    }


    public double[][] transformDatasetTopKWithBMU(List<double[]> dataset, int k) {
        List<int[]> active = getActiveNeuronIndices();
        SOMNeuron[][] neurons2D = som.getNeurons();
        double[][] output = new double[dataset.size()][k + 2];
        double rowDen = Math.max(1.0, som.getR() - 1.0);
        double colDen = Math.max(1.0, som.getC() - 1.0);
        for (int i = 0; i < dataset.size(); i++) {
            double[] input = dataset.get(i);
            // 1. BMU
            int[] bmu = som.findBMUCoords(input);
            double rowNorm = bmu[0] / rowDen;
            double colNorm = bmu[1] / colDen;
            output[i][0] = rowNorm;
            output[i][1] = colNorm;
            // 2. top-k distances
            double[] distances = new double[active.size()];
            for (int j = 0; j < active.size(); j++) {
                int[] idx = active.get(j);
                distances[j] = neurons2D[idx[0]][idx[1]].findDist(input, distance);
            }
            Arrays.sort(distances);
            int copyLen = Math.min(k, distances.length);
            System.arraycopy(distances, 0, output[i], 2, copyLen);
            if (copyLen > 0 && copyLen < k) {
                Arrays.fill(output[i], 2 + copyLen, 2 + k, distances[copyLen - 1]);
            }
        }
        return output;
    }

    public void printNeuronSummary() {
        System.out.println("\n=== SOM Neuron Summary ===");
        int inactive = 0;
        int mixed = 0;
        for (int i = 0; i < som.getR(); i++) {
            for (int j = 0; j < som.getC(); j++) {
                SOMNeuron neuron = som.getNeurons()[i][j];
                String className = getClassName(neuron.getlIndex());
                double purity = neuron.getNeuronalPurity();
                System.out.printf(
                        "Neuron [%02d,%02d]: label=%s, wins=%d, purity=%.3f, votes=%s%n",
                        i, j, className, neuron.getWinCount(), purity, neuron.getLabelCounts()
                );
                if (neuron.getWinCount() == 0) inactive++;
                if (neuron.getlIndex() == SOMNeuron.mix) mixed++;
            }
        }

        System.out.println("Inactive (dead) neurons: " + inactive);
        System.out.println("Mixed neurons: " + mixed);
    }

    private String getClassName(int idx) {
        if (idx == -2) return "Mixed";
        if (idx == -1) return "Unlabeled";
        return ClassLabels.getName(idx);
    }

    public SOM getSom() {
        return som;
    }

    public SOMNeuron getNeuron(int row, int col) {
        return som.getNeurons()[row][col];
    }

}