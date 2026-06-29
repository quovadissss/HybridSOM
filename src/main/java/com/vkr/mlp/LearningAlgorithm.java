package com.vkr.mlp;

public interface LearningAlgorithm {

    void updateW(MLPNeuron neuron, double learningRate, double norm);
}
