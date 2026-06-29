package com.vkr.mlp;

import java.util.Arrays;
import java.util.Random;

/**
 * Класс для описания слоя MLP,
 * состоящего из массива нейронов
 */
public class MLPLayer {
    private final MLPNeuron[] neurons;//массив нейронов
    private final boolean isOut;//булевское значение для проверки на выходной слой
    private boolean useSoftmax;//булевское значения для использования софтмакс

    /**
     * Конструктор слоя MLP
     * @param numN количество нейронов слоя
     * @param size количество входов каждого нейрона
     * @param isOut признак выходного слоя
     * @param activationFunc название функции активации
     * @param useSoftmax использование softmax на выходном слое
     * @param seed фиксированное значение для random
     */
    public MLPLayer(int numN, int size, boolean isOut, String activationFunc, boolean useSoftmax, long seed) {
        Random layerRand = new Random(seed);
        this.neurons = new MLPNeuron[numN];
        this.isOut = isOut;
        this.useSoftmax = useSoftmax;
        ActivationFunction af;
        if (isOut && !useSoftmax) {//если бинарная классификация - на выходе сигмоида
            af = new SigmoidActivation();
        } else if (activationFunc.equals("relu")) {//если слой скрытый - то функция relu
            af = new RELUActivation();
        } else {
            af = new SigmoidActivation();//иначе сигмоида
        }
        // создание нейронов слоя
        for (int i = 0; i < numN; i++) {
            neurons[i] = new MLPNeuron(size, af, layerRand);
        }
    }
    /**
     * Расчет выходных значений
     * @param inputs входные данные
     * @return выходы
     */
    public double[] calcOutput (double[] inputs) {
        double[] outputs = new double[neurons.length];
        if (isOut && useSoftmax) { // если выходной слой и софтмакс
            double[] l = new double[neurons.length];
            // вычисляем линейные выходы каждого нейрона: y = w*x + b
            for (int i = 0; i < neurons.length; i++) {
                MLPNeuron neuron = neurons[i];
                double y = neuron.getB(); // получаем смещение bias
                for (int j = 0; j < inputs.length; j++) {
                    y += inputs[j] *  neuron.getW()[j]; // w*x + b (вес * вход + смещение)
                }
                neuron.setLinOut(y); // сохраняем линейный выход нейрона
                l[i] = y;             // запоминаем для применения softmax
            }
            // применяем softmax
            double maxL = Arrays.stream(l).max().orElse(0); // для численной стабильности
            double sumExp = 0.0;
            // вычисляем экспоненты
            for (int i = 0; i < neurons.length; i++) {
                outputs[i] = Math.exp(l[i] - maxL); // exp с вычитанием max
                sumExp += outputs[i];
            }
            // нормализуем, чтобы получить вероятности (сумма = 1)
            for (int i = 0; i < neurons.length; i++) {
                outputs[i] /= sumExp;              // делим на сумму, чтобы получить вероятность
                neurons[i].setOutputWithActivationFunc(outputs[i]);  // сохраняем выход нейрона (y_i после softmax)
            }
        }
        else {
            // для скрытых слоев используем обычную активацию (relu)
            for (int i = 0; i < neurons.length; i++) {
                outputs[i] = neurons[i].computeOutput(inputs);//расчет выхода
            }
        }

        return outputs;
    }

    /**
     * Очистка накопленных градиентов всех нейронов слоя
     */
    public void clearGrad() {
        for (MLPNeuron n : neurons) {
            n.clearBatchGradients();//очищаем
        }
    }

    /**
     * Накопление градиентов нейронов слоя
     * Вычисленные значения локальной ошибки нейрона
     * используются для накопления градиентов весов и смещений
     * @param out выходы предыдущего слоя
     */
    public void accumulateGradientsFromNeurons(double[] out) {
        for (MLPNeuron n : neurons) {
            n.addGrad(n.getErr(), out);
        }
    }

    /**
     * Получение массива нейронов слоя.
     * @return массив нейронов
     */
    public MLPNeuron[] getNeurons() {
        return neurons;
    }

    /**
     * Проверка, является ли слой выходным.
     * @return true, если слой выходной
     */
    public boolean isOut() {
        return isOut;
    }

    /**
     * Проверка использования софтмакс
     * @return true, если для слоя используется softmax
     */
    public boolean isUseSoftmax() {
        return useSoftmax;
    }

}