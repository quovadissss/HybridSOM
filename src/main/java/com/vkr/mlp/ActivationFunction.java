package com.vkr.mlp;

public interface ActivationFunction {
    double activate(double x);
    double derivative(double x);
}