package com.vkr.mlp;

public class Adam implements LearningAlgorithm {

    private final int timeStep;

    public Adam(int timeStep) {
        this.timeStep = timeStep;
    }

    @Override
    public void updateW(
            MLPNeuron neuron,
            double learningRate,
            double norm
    ) {

        double[] weights = neuron.getW();

        double[] firstMoment =
                neuron.getAdamFirstMomentWeights();

        double[] secondMoment =
                neuron.getAdamSecondMomentWeights();

        double beta1 = 0.9;
        double beta2 = 0.999;

        double epsilon = 1e-8;

        for (int i = 0; i < weights.length; i++) {

            double gradient =
                    neuron.getWeightGradientSum()[i] * norm;

            firstMoment[i] =
                    beta1 * firstMoment[i]
                            + (1 - beta1) * gradient;

            secondMoment[i] =
                    beta2 * secondMoment[i]
                            + (1 - beta2) * gradient * gradient;

            double correctedFirstMoment =
                    firstMoment[i]
                            / (1.0 - Math.pow(beta1, timeStep));

            double correctedSecondMoment =
                    secondMoment[i]
                            / (1.0 - Math.pow(beta2, timeStep));

            double update =
                    learningRate
                            * correctedFirstMoment
                            / (Math.sqrt(correctedSecondMoment)
                            + epsilon);

            weights[i] -= update;
        }

        neuron.setW(weights);

        neuron.setAdamFirstMomentWeights(firstMoment);

        neuron.setAdamSecondMomentWeights(secondMoment);

        double biasGradient =
                neuron.getBiasGradientSum() * norm;

        double firstMomentBias =
                neuron.getAdamFirstMomentBias();

        double secondMomentBias =
                neuron.getAdamSecondMomentBias();

        firstMomentBias =
                beta1 * firstMomentBias
                        + (1 - beta1) * biasGradient;

        secondMomentBias =
                beta2 * secondMomentBias
                        + (1 - beta2)
                        * biasGradient
                        * biasGradient;

        double correctedFirstMomentBias =
                firstMomentBias
                        / (1.0 - Math.pow(beta1, timeStep));

        double correctedSecondMomentBias =
                secondMomentBias
                        / (1.0 - Math.pow(beta2, timeStep));

        double biasUpdate =
                learningRate
                        * correctedFirstMomentBias
                        / (Math.sqrt(correctedSecondMomentBias)
                        + epsilon);

        neuron.setB(
                neuron.getB() - biasUpdate
        );

        neuron.setAdamFirstMomentBias(firstMomentBias);

        neuron.setAdamSecondMomentBias(secondMomentBias);
    }
}