package com.vkr.som;
import java.util.*;

public class SOM {

    private final SOMNeuron[][] neurons; //решетка из нейронов
    private final int r, c; //число строк и столбцов решетки
    private final String dst; //мера расстояния
    private final Random shuffleRand;//random для перемешивания
    private final Random jitterRand;//random для добавления шума
    private double qe = 0.0;//ошибка квантования

    /**
     * Конструктор SOM
     * @param r количество строк карты
     * @param c количество столбцов карты
     * @param inputData обучающие данные
     * @param distance мера расстояния
     * @param seed фикс. значение для всех random
     */
    public SOM(int r, int c, List<double[]> inputData, String distance, long seed) {
        this.r = r;
        this.c = c;
        this.dst = distance;
        Random initRand = new Random(seed);
        this.shuffleRand = new Random(seed+1);
        this.jitterRand = new Random(seed+2);
        System.out.println("Инициализация SOM " + r + "x" + c);
        neurons = new SOMNeuron[r][c];
        int d = inputData.get(0).length;//определяем размер признаков
        double[] min = new double[d]; //массив минимумов
        double[] max = new double[d];//массив максимумов
        //заполняем массивы
        Arrays.fill(min, Double.POSITIVE_INFINITY);
        Arrays.fill(max, Double.NEGATIVE_INFINITY);
        for (double[] v : inputData) {
            for (int i = 0; i < d; i++) {
                min[i] = Math.min(min[i], v[i]);//поиск минимальных
                max[i] = Math.max(max[i], v[i]);//и максимальных значений
            }
        }
        //инициализация весов
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                double[] w = new double[d];//создаем весовой вектор
                for (int k = 0; k < d; k++) {
                    //случайная генерация весов на основе диапазона обучающей выборки
                    w[k] = min[k] + initRand.nextDouble() * (max[k] - min[k]);
                }
                neurons[i][j] = new SOMNeuron(w);
            }
        }
    }



    /**
     * Запуск двухфазного обучение SOM (грубая и тонкая настройка)
     * @param data обучающие данные
     * @param labels индексы классов для циклической подачи
     * @param iter1 количество итераций фазы 1
     * @param iter2 количество итераций фазы 2
     * @param lr1 начальный learning rate фазы 1
     * @param lr2 начальный learning rate фазы 2
     * @param r1 начальный радиус фазы 1
     * @param r2 начальный радиус фазы 2
     */
    public void trainTwoPhaseBalanced(List<double[]> data,
                                      List<Integer> labels,
                                      int iter1,
                                      int iter2,
                                      double lr1,
                                      double lr2,
                                      double r1,
                                      double r2) {
        System.out.println("\nФаза 1 (грубая)");
        trainPhaseBalanced(data, labels, iter1, lr1, r1, 1.5);//запуск грубой фазы
        System.out.println("\nФаза 2 (тонкая)");
        trainPhaseBalanced(data, labels, iter2, lr2, r2, 1.0);//запуск тонкой фазы
        qe = calculateQE(data);//расчет ошибки квантования
        System.out.println("QE финальное = " + qe);
    }

    /**
     * Запуск одной фазы обучения SOM с циклической подачей классов
     * @param data обучающие данные
     * @param labels индексы классов
     * @param n количество итераций
     * @param lrStart начальный learning rate
     * @param r начальный радиус соседства
     * @param minR минимальный радиус
     */
    private void trainPhaseBalanced(List<double[]> data,
                                    List<Integer> labels,
                                    int n,
                                    double lrStart,
                                    double r,
                                    double minR) {
        Map<Integer, List<double[]>> groupedSamples = createClassGroup(data, labels);//получаем группы
        List<Integer> classes = new ArrayList<>(groupedSamples.keySet()); //получаем список классов
        Collections.shuffle(classes, shuffleRand);//перемешиваем список
        double tau = n / Math.log(Math.max(r, 1.0001)); //параметр управления скоростью уменьшения радиуса
        resetWinCounts(); //сброс побед нейронов
        for (int t = 0; t < n; t++) {//проход по итерациям
            double lr = lrStart * Math.exp(-t / (double) n);//экспоненциальное затухание lr
            double radius = Math.max(r * Math.exp(-t / tau), minR); //затухание радиуса
            int cls = classes.get(t % classes.size());//находим индекс класса
            List<double[]> classSamples = groupedSamples.get(cls);//и получаем его образцы
            double[] input = classSamples
                    .get(shuffleRand.nextInt(classSamples.size()))
                    .clone();//выбираем случайны образец и клонируем его
            input = SOMUtilities.addTinyJitter(input, jitterRand); //добавляем шум
            int[] bmuNeuron = findBMUCoords(input); //находим BMU
            neurons[bmuNeuron[0]][bmuNeuron[1]].incWinCount();//обновляем счетчик побед нейрона
            for (int i = 0; i < this.r; i++) {
                for (int j = 0; j < c; j++) {
                    //обновление соседей
                    double dstSq = (i - bmuNeuron[0]) * (i - bmuNeuron[0])
                                    + (j - bmuNeuron[1]) * (j - bmuNeuron[1]);
                    if (dstSq > radius * radius) continue; //проверка на попадание в радиус
                    double infl = SOMUtilities.gaussian(dstSq, radius);//вычисляем влияние
                    neurons[i][j].adjustment(input, lr * infl);//корректируем веса
                }
            }
        }
    }
    /**
     * Создание групп данных по классам
     * @param data обучающие данные
     * @param labels индексы классов
     * @return хранилище индекс класса - объекты класса
     */
    private Map<Integer, List<double[]>> createClassGroup(List<double[]> data, List<Integer> labels) {
        Map<Integer, List<double[]>> groupedSamples = new HashMap<>();
        for (int ind = 0; ind < data.size(); ind++) { //проход по выборке
            int currentLabel = labels.get(ind); //получение текущего индекса класса
            if (!groupedSamples.containsKey(currentLabel)) { //проверка, содержит ли хранилище данную метку
                groupedSamples.put(currentLabel, new ArrayList<>());
            }
            groupedSamples.get(currentLabel).add(data.get(ind));//добавляем образец
        }
        return groupedSamples ;
    }

    /**
     * Поиск победителя
     * @param input входной вектор
     * @return координаты победителя
     */
    public int[] findBMUCoords(double[] input) {
        int resR = 0;
        int resC = 0;
        double minDist = neurons[0][0].findDist(input, dst);//начальное min значение
        for (int rows = 0; rows < this.r; rows++) {
            for (int cols = 0; cols < this.c; cols++) {
                double dist = neurons[rows][cols].findDist(input, dst);//находим расстояние
                if (dist < minDist) { //если расстояние < минимального
                    minDist = dist;//записываем новое минимальное значение
                    resR = rows;//запоминаем
                    resC = cols;//координаты
                }
            }
        }
        return new int[]{resR, resC};
    }

    /**
     * Расчет ошибки квантования QE
     * @param data входные данные
     * @return ошибка квантования
     */
    public double calculateQE(List<double[]> data) {
        double err = 0.0;//накапливаемая ошибка
        for (double[] d : data) {
            int[] w = findBMUCoords(d);//находим координаты
            err += neurons[w[0]][w[1]].findDist(d, dst);//накапливаем расстояния
        }
        return err / data.size();
    }
    /**
     * Сброс счетчиков побед всех нейронов
     */
    public void resetWinCounts() {
        for (int r = 0; r < this.r; r++) {
            for (int c = 0; c < this.c; c++) {
                neurons[r][c].resetWinCount();//сбрасываем счетчик побед
            }
        }
    }

    /**
     * Сброс голосов и меток нейронов
     */
    public void resetLabels() {
        for (int r = 0; r < this.r; r++) {
            for (int c = 0; c < this.c; c++) {
                neurons[r][c].clearLabelCounts();//очищаем метки
                neurons[r][c].setlIndex(-1);
            }
        }
    }


    /**
     * Маркировка нейронов на основе выборки для обучения
     * @param data входные векторы
     * @param labels метки в one-hot формате
     * @param purityBoundary граничное значение чистоты нейрона
     */
    public void assignNeuronLabels(List<double[]> data, List<double[]> labels, double purityBoundary) {
        resetLabels();//сброс старых меток
        for (int i = 0; i < data.size(); i++) {
            int[] w = findBMUCoords(data.get(i));//находим координаты
            SOMNeuron neuron = neurons[w[0]][w[1]];//победителя
            neuron.addLabel(SOMUtilities.argmax(labels.get(i)));//добавляем информацию о классе объекта в статистику нейрона
        }
        for (int r = 0; r < this.r; r++) {
            for (int c = 0; c < this.c; c++) {
                neurons[r][c].defineLabel(purityBoundary);//определяем итоговую метку нейрона
            }
        }
    }


    /**
     * Получение массива нейронов
     * @return нейроны карты
     */
    public SOMNeuron[][] getNeurons() { return neurons; }

    /**
     * Получение значения QE
     * @return значение QE
     */
    public double getQe() { return qe; }

    /**
     * Получение числа строк
     * @return число строк
     */
    public int getR() { return r;}

    /**
     * Получение числа колонок
     * @return число колонок
     */
    public int getC() { return c; }
    /**
     * Получение меры расстояния
     * @return мера расстояния
     */
    public String getDst() { return dst; }

    /**
     * Конструктор SOM на основе готового массива нейронов.
     *
     * @param rows количество строк карты
     * @param cols количество столбцов карты
     * @param neurons массив нейронов
     * @param distance используемая мера расстояния
     * @param qe значение ошибки квантования
     * @param seed фиксированное значение для random
     */
    public SOM(int rows,
               int cols,
               SOMNeuron[][] neurons,
               String distance,
               double qe,
               long seed) {

        this.r = rows;
        this.c = cols;
        this.dst = distance;
        this.neurons = neurons;
        this.qe = qe;

        this.shuffleRand = new Random(seed + 1);
        this.jitterRand = new Random(seed + 2);
    }

    /**
     * Расчёт U-matrix — средней дистанции
     * между весами нейрона и его соседей.
     *
     * Используются только 4 соседа:
     * сверху, снизу, слева и справа.
     *
     * @return матрица U размером rows x cols
     */
    public double[][] computeUMatrix() {

        double[][] u = new double[r][c];

        int[][] dirs = {
                {-1, 0},
                {0, -1},
                {0, 1},
                {1, 0}
        };

        for (int r = 0; r < this.r; r++) {

            for (int c = 0; c < this.c; c++) {

                double sum = 0.0;
                int count = 0;

                for (int[] d : dirs) {

                    int nr = r + d[0];
                    int nc = c + d[1];

                    if (nr >= 0 &&
                            nr < this.r &&
                            nc >= 0 &&
                            nc < this.c) {

                        sum += SOMUtilities.euclidean(
                                neurons[r][c].getWeights(),
                                neurons[nr][nc].getWeights()
                        );

                        count++;
                    }
                }

                u[r][c] = count > 0
                        ? sum / count
                        : 0.0;
            }
        }

        return u;
    }
    /**
     * Подсчёт количества побед нейронов
     * на заданном наборе данных.
     *
     * @param data входные данные
     */
    public void countWinnersOnDataset(double[][] data) {
        resetWinCounts();
        for (double[] d : data) {
            int[] w = findBMUCoords(d);
            neurons[w[0]][w[1]].incWinCount();//накапливаем победы
        }
        SOMUtilities.printWinStatistics(neurons, r, c);//печатаем статистику
    }
}

