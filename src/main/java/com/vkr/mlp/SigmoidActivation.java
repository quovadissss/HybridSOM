package com.vkr.mlp;

class SigmoidActivation implements ActivationFunction {
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double derivative(double x) {
        double s = activate(x);
        return s * (1.0 - s);
    }
}