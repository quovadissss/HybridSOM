package com.vkr.mlp;

public class SGD implements LearningAlgorithm {

    private final double momentum;

    public SGD(double momentum) {
        this.momentum = momentum;
    }

    @Override
    public void updateW(
            MLPNeuron neuron,
            double learningRate,
            double norm
    ) {

        double[] weights = neuron.getW();

        double[] velocityWeights =
                neuron.getMomentumWeights();

        for (int i = 0; i < weights.length; i++) {

            double gradient =
                    neuron.getWeightGradientSum()[i] * norm;

            double newVelocity =
                    momentum * velocityWeights[i]
                            + learningRate * gradient;

            velocityWeights[i] = newVelocity;

            weights[i] -= newVelocity;
        }

        neuron.setW(weights);

        neuron.setMomentumWeights(velocityWeights);

        double biasGradient =
                neuron.getBiasGradientSum() * norm;

        double biasVelocity =
                neuron.getMomentumBias();

        double newBiasVelocity =
                momentum * biasVelocity
                        + learningRate * biasGradient;

        neuron.setMomentumBias(newBiasVelocity);

        neuron.setB(
                neuron.getB() - newBiasVelocity
        );
    }
}