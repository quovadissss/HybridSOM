package com.vkr.mlp;

import java.util.Arrays;
import java.util.Random;

/**
 * Класс, описывающий отдельный нейрон многослойного персептрона.
 * Нейрон хранит веса, смещение, результат прямого прохода,
 * локальную ошибку, а также служебные параметры для оптимизаторов.
 */
public class MLPNeuron {

    private double[] w;// веса связей с предыдущим слоем
    private double b;// смещение нейрона
    private double outputWithActivationFunc; // сигнал на выходе после функции активации
    private double linOut;// сигнал до функции активации
    private double err;// локальная ошибка нейрона
    private double[] momentumWeights;// скорости изменения весов для momentum
    private double momentumBias;// скорость изменения bias для momentum
    private double[] adamFirstMomentWeights; // первый момент весов для Adam
    private double[] adamSecondMomentWeights;// второй момент весов для Adam
    private double adamFirstMomentBias;// первый момент bias для Adam
    private double adamSecondMomentBias;// второй момент bias для Adam
    private double[] weightGradientSum;//градиенты весов за mini-batch
    private double biasGradientSum;// суммарный градиент смещения за mini-batch
    private final ActivationFunction activationFunction;//функция активации

    /**
     * Инициализация одного нейрона
     * @param size количество входов нейрона
     * @param activationFunction функция активации
     * @param rand генератор случайных чисел
     */
    public MLPNeuron(
            int size,
            ActivationFunction activationFunction,
            Random rand
    ) {
        //инициализация полей
        this.w = new double[size];
        this.momentumWeights = new double[size];
        this.adamFirstMomentWeights = new double[size];
        this.adamSecondMomentWeights = new double[size];
        this.weightGradientSum = new double[size];
        double scale = Math.sqrt(2.0 / size);//вычисляем масштаб инициализации
        for (int i = 0; i < size; i++) {
            w[i] = rand.nextGaussian() * scale;//He initialization
            momentumWeights[i] = 0.0;
            adamFirstMomentWeights[i] = 0.0;
            adamSecondMomentWeights[i] = 0.0;
            weightGradientSum[i] = 0.0;
        }
        this.b = 0.01;
        this.momentumBias = 0.0;
        this.adamFirstMomentBias = 0.0;
        this.adamSecondMomentBias = 0.0;
        this.biasGradientSum = 0.0;
        this.activationFunction = activationFunction;
    }

    /**
     * Этап прямого прохода
     * @param input значения с предыдущего слоя
     * @return выходной сигнал после функции активации
     */
    public double computeOutput(double[] input) {
        double sum = b;
        for (int i = 0; i < input.length; i++) {
            sum += input[i] * w[i];
        }
        this.linOut = sum;
        this.outputWithActivationFunc = activationFunction.activate(sum);
        return outputWithActivationFunc;
    }

    /**
     * Сброс накопленных градиентов перед обработкой нового мини-пакета
     */
    public void clearBatchGradients() {
        Arrays.fill(weightGradientSum, 0.0);
        biasGradientSum = 0.0;
    }

    /**
     * Накопление градиентов по одному обучающему примеру
     * @param err локальная ошибка нейрона
     * @param previousLayerOutputs выходы предыдущего слоя
     */
    public void addGrad(double err, double[] previousLayerOutputs) {
        for (int i = 0; i < w.length; i++) {
            weightGradientSum[i] += err * previousLayerOutputs[i];
        }
        biasGradientSum += err;
    }

    public double[] getW() {
        return w;
    }

    public void setW(double[] w) {
        this.w = w;
    }

    public double getB() {
        return b;
    }

    public void setB(double b) {
        this.b = b;
    }

    public double getOutputWithActivationFunc() {
        return outputWithActivationFunc;
    }


    public void setOutputWithActivationFunc(double outputWithActivationFunc) {
        this.outputWithActivationFunc = outputWithActivationFunc;
    }

    public double getLinOut() {
        return linOut;
    }

    public void setLinOut(double linOut) {
        this.linOut = linOut;
    }

    public double getErr() {
        return err;
    }

    public void setErr(double err) {
        this.err = err;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public double[] getMomentumWeights() {
        return momentumWeights;
    }

    public void setMomentumWeights(double[] momentumWeights) {
        this.momentumWeights = momentumWeights;
    }

    public double getMomentumBias() {
        return momentumBias;
    }

    public void setMomentumBias(double momentumBias) {
        this.momentumBias = momentumBias;
    }

    public double[] getAdamFirstMomentWeights() {
        return adamFirstMomentWeights;
    }

    public void setAdamFirstMomentWeights(double[] adamFirstMomentWeights) {
        this.adamFirstMomentWeights = adamFirstMomentWeights;
    }

    public double[] getAdamSecondMomentWeights() {
        return adamSecondMomentWeights;
    }

    public void setAdamSecondMomentWeights(double[] adamSecondMomentWeights) {
        this.adamSecondMomentWeights = adamSecondMomentWeights;
    }

    public double getAdamFirstMomentBias() {
        return adamFirstMomentBias;
    }

    public void setAdamFirstMomentBias(double adamFirstMomentBias) {
        this.adamFirstMomentBias = adamFirstMomentBias;
    }

    public double getAdamSecondMomentBias() {
        return adamSecondMomentBias;
    }

    public void setAdamSecondMomentBias(double adamSecondMomentBias) {
        this.adamSecondMomentBias = adamSecondMomentBias;
    }

    public double[] getWeightGradientSum() {
        return weightGradientSum;
    }

    public double getBiasGradientSum() {
        return biasGradientSum;
    }
}