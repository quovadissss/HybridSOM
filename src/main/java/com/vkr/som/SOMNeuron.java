package com.vkr.som;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Класс для узла решетки SOM.
 */
public class SOMNeuron {

    public static final int mix = -2;//метка, которой маркируется нейрон при его смешанности
    private final double[] w;//массив весов
    private double meanWinDistance = 0.0;//среднее расстояние от экземпляров, для которых этот узел - BMU
    private double stdFromObj = 0.0; //отклонение расстояние от экземпляров, для которых этот узел - BMU
    private int lIndex = -1;//итоговая метка нейрона
    private int winCount = 0;//кол-во побед
    private final Map<Integer, Integer> labelCounts = new HashMap<>(); //хэш мап для подсчета числа классов

    /**
     * Инициализация узла
     * @param initValues начальные значения весов
     */
    public SOMNeuron(double[] initValues) {
        this.w = Arrays.copyOf(initValues, initValues.length);
    }

    /**
     * Расчет дистанции между образцом и узлом.
     *
     * @param x входной вектор
     * @param funkDist используемая мера расстояния
     * @return расстояние до узла
     */
    public double findDist(double[] x, String funkDist) {
        if (!Objects.equals(funkDist, "euclidean")) {
            throw new IllegalArgumentException("Invalid distance " + funkDist);
        }
        return SOMUtilities.euclidean(x, w);
    }

    /**
     * Изменение весовых коэффициентов
     * @param x входной вектор
     * @param l коэффициент обучения
     */
    public void adjustment(double[] x, double l) {
        int h = 0;
        while (h < w.length) {
            w[h] += l * (x[h] - w[h]);//стандартное правило корректировки
            h++;
        }
    }

    /**
     * Добавление метки класса, если нейрон победил для объекта этого класса
     * @param label индекс класса
     */
    public void addLabel(int label) {
        labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
    }

    /**
     * Очистка накопленных меток классов
     */
    public void clearLabelCounts() {
        labelCounts.clear();
    }

    /**
     * Получение количества объектов каждого класса,
     * попавших в данный нейрон.
     * @return хранилище индекс класса - число объектов этого класса
     */
    public Map<Integer, Integer> getLabelCounts() {
        return labelCounts;
    }

    /**
     * Получение общего количества меток
     * @return суммарное число объектов,
     * попавших в данный нейрон
     */
    public int getTotalLabels() {
        return labelCounts.values()
                .stream()
                .mapToInt(Integer::intValue)
                .sum();
    }

    /**
     * Вычисление показателя чистоты нейрона
     * @return значение чистоты в диапазоне от 0 до 1
     */
    public double getNeuronalPurity() {
        if(getTotalLabels() == 0) return 0;
        int maxCount = 0;
        for (Integer count : labelCounts.values()) {
            if (count > maxCount) {
                maxCount = count;
            }
        }
        return maxCount / (double) getTotalLabels();
    }

    /**
     * Определение итоговой метки нейрона с учетом границы чистоты
     * @param neuronPurityBoundary граница чистоты
     */
    public void defineLabel(double neuronPurityBoundary) {
        if (labelCounts.isEmpty()) {
            lIndex = -1;
            return;
        }
        Map.Entry<Integer, Integer> dominantClass = null;
        for (Map.Entry<Integer, Integer> entry : labelCounts.entrySet()) {
            //поиск доминирующего класса
            if (dominantClass == null || entry.getValue() > dominantClass.getValue()) {
                dominantClass = entry;
            }
        }
        if (dominantClass == null) {
            lIndex = -1;
            return;
        }
        if (dominantClass.getValue() / (double) getTotalLabels() >= neuronPurityBoundary) {
            lIndex = dominantClass.getKey();
        } else {
            lIndex = mix;
        }
    }

    /**
     * Получение среднего расстояния до нейрона.
     * @return среднее расстояние
     */
    public double getMeanWinDistance() {
        return meanWinDistance;
    }

    /**
     * Получение стандартного отклонения расстояний
     * @return стандартное отклонение
     */
    public double getStdFromObj() {
        return stdFromObj;
    }

    /**
     * Установка количества побед нейрона
     * @param winCount число побед
     */
    public void setWinCount(int winCount) {
        this.winCount = winCount;
    }

    /**
     * Добавление меток класса
     * @param label индекс класса
     * @param count количество голосов
     */
    public void putLabelCount(int label, int count) {
        labelCounts.put(label, count);
    }

    /**
     * Установка статистических параметров
     * Используется для слоя принятия решений
     * @param meanWinDistance среднее расстояние
     * @param stdWinDistance стандартное отклонение
     */
    public void setDistanceStats(
            double meanWinDistance,
            double stdWinDistance
    ) {
        this.meanWinDistance = meanWinDistance;
        this.stdFromObj = stdWinDistance;
    }

    /**
     * Увеличение счетчика побед нейрона
     */
    public void incWinCount() {
        winCount++;
    }

    /**
     * Сброс количества побед нейрона
     */
    public void resetWinCount() {
        this.winCount = 0;
    }

    /**
     * Получение метки нейрона
     * @return индекс класса
     */
    public int getlIndex() {
        return lIndex;
    }

    /**
     * Установка итоговой метки нейрона.
     * @param lIndex индекс класса
     */
    public void setlIndex(int lIndex) {
        this.lIndex = lIndex;
    }

    /**
     * Получение количества побед нейрона
     * @return число побед
     */
    public int getWinCount() {
        return winCount;
    }

    /**
     * Получение весов нейрона
     * @return вектор весов
     */
    public double[] getWeights() {
        return w;
    }
}