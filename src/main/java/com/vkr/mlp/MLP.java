package com.vkr.mlp;

import java.util.ArrayList;
import java.util.List;

/**
 * Класс, реализующий MLP,
 * состоящий из слоев MLPLayers
 */
public class MLP {

    private final MLPLayer[] layers;//массив слоев
    private final double l;//коэфф обучения
    private final double m;//значение момента для SGD с моментом
    private final String algorithm;//алгоритм обучения
    private int timeStep = 0;

    /**
     * Создание многослойного персептрона
     * @param layerSizes размеры слоев сети
     * @param l коэффициент обучения
     * @param moment значение момента для SGD с моментом
     * @param activationFunc функция активации
     * @param algorithm алгоритм обучения
     * @param seed фиксированное значение для random
     */
    public MLP(int[] layerSizes, double l, double moment, String activationFunc,
               String algorithm, long seed) {
        //инициализация
        this.l = l;
        this.m = moment;
        this.algorithm = algorithm.toLowerCase();
        this.layers = new MLPLayer[layerSizes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            //проверка, является ли текущий слой последним
            boolean outputLayer = i == layers.length - 1;
            //софтмакс используется только на выходном слое, если выходных нейронов > 1
            boolean softmaxRequired = outputLayer && layerSizes[layerSizes.length - 1] > 1;
            layers[i] = new MLPLayer(
                    layerSizes[i + 1],//количество нейронов текущего слоя
                    layerSizes[i],//количество входов каждого нейрона текущего слоя
                    outputLayer,
                    activationFunc,
                    softmaxRequired,
                    seed
            );
        }
    }

    /**
     * Осуществление прямого прохода с вычислением предсказания
     * @param data входной вектор признаков
     * @return выходной вектор сети
     */
    public double[] getPrediction(double[] data) {
        double[] currentOutput = data.clone();//клонируем, чтобы не менять исходные данные
        int ind = 0;
        while (ind < layers.length) {
            currentOutput = layers[ind].calcOutput(currentOutput);
            ind++;
        }
        return currentOutput;
    }

    /**
     * Осуществление обратного прохода по сети
     * @param target ожидаемый выход сети
     */
    private void calculateBackpropagationErrors(double[] target) {
        MLPLayer outputLayer = layers[layers.length - 1];//берем выходной слой
        for (int i = 0; i < outputLayer.getNeurons().length; i++) {
            //находим реальный выход
            double real = outputLayer.getNeurons()[i].getOutputWithActivationFunc();
            //вычисляем и присваиваем ошибку
            outputLayer.getNeurons()[i].setErr(real - target[i]);
        }
        for (int ind = layers.length - 2; ind >= 0; ind--) {//для скрытых слоев обратным проходом
            MLPLayer curr = layers[ind];//текущий
            MLPLayer next = layers[ind + 1];//и следующий слой
            //Проход по нейронам текущего слоя
            int neuronInd = 0;
            while(neuronInd<curr.getNeurons().length) {
                double errSum = 0;//накопление ошибки от следующего слоя
                for (MLPNeuron nextNeuron : next.getNeurons()) {
                    //+ вклад ошибки следующего слоя через вес связи
                    errSum += nextNeuron.getErr() * nextNeuron.getW()[neuronInd];
                }
                //вычисление производной
                double derivative = curr.getNeurons()[neuronInd].getActivationFunction()
                        .derivative(curr.getNeurons()[neuronInd].getLinOut());
                curr.getNeurons()[neuronInd].setErr(errSum * derivative);//записываем ошибку текущего нейрона
                neuronInd++;
            }
        }
    }

    /**
     * Очистка накопленных градиентов перед обработкой следующего мини-пакета
     */
    public void clearBatchGradients() {
        int ind = 0;
        while (ind < layers.length) {
            layers[ind].clearGrad();
            ind++;
        }
    }

    /**
     * Выполнение прямого прохода, обратного распространения ошибки
     * и накопления градиентов по одному экземпляру из обучающих данных
     * @param input входной вектор признаков
     * @param target ожидаемый выход сети
     * @return фактический выход сети
     */
    public double[] processTrainingExample(double[] input, double[] target) {
        List<double[]> layerOutputs = new ArrayList<>();//список для хранения выходов всех слоев
        double[] curr = input.clone();//клонируем входной вектор
        for (MLPLayer lrs : layers) {
            curr = lrs.calcOutput(curr);//считаем выходы через прямой проход по сети
            layerOutputs.add(curr);//сохраняем
        }
        calculateBackpropagationErrors(target);//считаем ошибки
        for (int lrInd = 0; lrInd < layers.length; lrInd++) {
            double[] prev;
            if (lrInd == 0) { // для первого слоя предыдущими значениями
                // являются исходные входные данные
                prev = input;
            }
            else {// для остальных слоев берем выход
                // предыдущего слоя сети
                prev = layerOutputs.get(lrInd - 1);
            }
            // накапливаем градиенты нейронов слоя
            layers[lrInd].accumulateGradientsFromNeurons(prev);
        }
        return curr;
    }

    /**
     * Применение накопленных градиентов после обработки мини-пакета
     * @param batchSize размер мини-пакета
     */
    public void updateParametersAfterBatch(int batchSize) {
        double k = 1.0 / batchSize;//коэфф усреднения градиентов
        timeStep++;//увеличиваем номер шага (для adam)
        LearningAlgorithm alg;
        if (algorithm.equals("adam")) {//если адам
            alg = new Adam(timeStep);
        } else {//иначе сгд
            alg = new SGD(m);
        }
        int layerInd = 0;
        while (layerInd < layers.length) {
            MLPNeuron[] neurons = layers[layerInd].getNeurons();
            int neuronInd = 0;
            while (neuronInd < neurons.length) {
                alg.updateW(neurons[neuronInd], l, k);
                neurons[neuronInd].clearBatchGradients();
                neuronInd++;
            }
            layerInd++;
        }
    }

    /**
     * Получение массива слоев MLP
     * @return слои MLP
     */
    public MLPLayer[] getLayers() {
        return layers;
    }

    /**
     * Получение названия алгоритма обучения
     * @return алгоритм обучения
     */
    public String getAlgorithm() {
        return algorithm;
    }

    /**
     * Получение коэффициента обучения
     * @return lr
     */
    public double getL() {
        return l;
    }

    /**
     * Получение момента
     * @return момент
     */
    public double getM() {
        return m;
    }
}