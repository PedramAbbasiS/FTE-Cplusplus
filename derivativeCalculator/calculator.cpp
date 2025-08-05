#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
using namespace std;

// Forward mode classes
class Function {
public:
    virtual double eval(double x) const = 0;
    virtual double derivative(double x) const = 0;
};

class Constant : public Function {
    double c;
public:
    Constant(double val) : c(val) {}
    double eval(double x) const override { return c; }
    double derivative(double x) const override { return 0.0; }
};

class Power : public Function {
    double n;
public:
    Power(double exponent) : n(exponent) {}
    double eval(double x) const override { return pow(x, n); }
    double derivative(double x) const override { return n * pow(x, n - 1); }
};

class Log : public Function {
    unique_ptr<Function> u;
public:
    Log(unique_ptr<Function> inner) : u(move(inner)) {}
    double eval(double x) const override {
        double val = u->eval(x);
        return log(val);
    }
    double derivative(double x) const override {
        double val = u->eval(x);
        return u->derivative(x) / val;
    }
};

class Add : public Function {
    unique_ptr<Function> function1, function2;
public:
    Add(unique_ptr<Function> left, unique_ptr<Function> right)
        : function1(move(left)), function2(move(right)) {
    }
    double eval(double x) const override { return function1->eval(x) + function2->eval(x); }
    double derivative(double x) const override { return function1->derivative(x) + function2->derivative(x); }
};

class Subtract : public Function {
    unique_ptr<Function> function1, function2;
public:
    Subtract(unique_ptr<Function> left, unique_ptr<Function> right)
        : function1(move(left)), function2(move(right)) {
    }
    double eval(double x) const override { return function1->eval(x) - function2->eval(x); }
    double derivative(double x) const override { return function1->derivative(x) - function2->derivative(x); }
};

class Product : public Function {
    unique_ptr<Function> function1, function2;
public:
    Product(unique_ptr<Function> left, unique_ptr<Function> right)
        : function1(move(left)), function2(move(right)) {
    }
    double eval(double x) const override { return function1->eval(x) * function2->eval(x); }
    double derivative(double x) const override {
        return function1->derivative(x) * function2->eval(x) + function1->eval(x) * function2->derivative(x);
    }
};

class Division : public Function {
    unique_ptr<Function> numerator, denominator;
public:
    Division(unique_ptr<Function> numerator, unique_ptr<Function> denominator)
        : numerator(move(numerator)), denominator(move(denominator)) {
    }
    double eval(double x) const override {
        double denom_val = denominator->eval(x);
        return numerator->eval(x) / denom_val;
    }
    double derivative(double x) const override {
        double denom_val = denominator->eval(x);
        double denom = denom_val * denom_val;
        return (numerator->derivative(x) * denom_val - numerator->eval(x) * denominator->derivative(x)) / denom;
    }
};

// Backward mode classes
class Node {
public:
    double value = 0.0;
    double grad = 0.0;
    virtual void forward() = 0;
    virtual void backward(double grad_in) = 0;
    virtual ~Node() {}
};

class InputNode : public Node {
public:
    InputNode(double v) { value = v; }
    void forward() override {}
    void backward(double grad_in) override { grad += grad_in; }
};

class ConstantNode : public Node {
public:
    ConstantNode(double v) { value = v; }
    void forward() override {}
    void backward(double grad_in) override {} // No gradient accumulation in case of constant node
};

class AdditionNode : public Node {
    Node* first, * second;
public:
    AdditionNode(Node* first_, Node* second_) : first(first_), second(second_) {}
    void forward() override { value = first->value + second->value; }
    void backward(double grad_in) override {
        first->backward(grad_in);
        second->backward(grad_in);
    }
};

class SubtractionNode : public Node {
    Node* first, * second;
public:
    SubtractionNode(Node* first_, Node* second_) : first(first_), second(second_) {}
    void forward() override { value = first->value - second->value; }
    void backward(double grad_in) override {
        first->backward(grad_in);
        second->backward(-grad_in);
    }
};

class ProductNode : public Node {
    Node* first, * second;
public:
    ProductNode(Node* first_, Node* second_) : first(first_), second(second_) {}
    void forward() override { value = first->value * second->value; }
    void backward(double grad_in) override {
        first->backward(grad_in * second->value);
        second->backward(grad_in * first->value);
    }
};

class DivisionNode : public Node {
    Node* numerator, * denominator;
public:
    DivisionNode(Node* numerator_, Node* denominator_) : numerator(numerator_), denominator(denominator_) {}
    void forward() override { value = numerator->value / denominator->value; }
    void backward(double grad_in) override {
        numerator->backward(grad_in / denominator->value);
        denominator->backward(-grad_in * numerator->value / (denominator->value * denominator->value));
    }
};

class PowerNode : public Node {
    Node* a;
    double n;
public:
    PowerNode(Node* base, double exp) : a(base), n(exp) {}
    void forward() override { value = pow(a->value, n); }
    void backward(double grad_in) override {
        if (a->value != 0)
            a->backward(grad_in * n * pow(a->value, n - 1));
    }
};

class LogNode : public Node {
    Node* a;
public:
    LogNode(Node* arg) : a(arg) {}
    void forward() override { value = log(a->value); }
    void backward(double grad_in) override {
        a->backward(grad_in / a->value);
    }
};

int main() {
    // x0:
    double x0 = 1.5;

    // function f and g definitions: Forward mode
    unique_ptr<Function> g = make_unique<Product>(
        make_unique<Subtract>(make_unique<Power>(2), make_unique<Constant>(5)),
        make_unique<Subtract>(make_unique<Constant>(4), make_unique<Product>(make_unique<Constant>(3), make_unique<Power>(1)))
    );

    // handle domain related erros
    if (g->eval(x0) <= 0) {
        cerr << "Domain Error: The value of the function inside ln (g(x)) negative and therfore the function and its derivative are not defined" << endl;
        exit(EXIT_FAILURE);
    }
    unique_ptr<Function> log_g = make_unique<Log>(move(g));
    unique_ptr<Function> log_g_divided_by_x_minus_4 = make_unique<Division>(
        move(log_g),
        make_unique<Subtract>(make_unique<Power>(1), make_unique<Constant>(4))
    );

    // handle domain related erros
    if (x0 == 4) {
        cerr << "Domain Error: X is equal to 4 and therfore the function and its derivative are not defined" << endl;
        exit(EXIT_FAILURE);
    }

    unique_ptr<Function> f = make_unique<Subtract>(
        make_unique<Add>(make_unique<Constant>(5), make_unique<Power>(3)),
        move(log_g_divided_by_x_minus_4)
    );

    //output the titles for results 
    cout << "x0\tf(x0)\tForward f'(x0)\tBackward f'(x0)" << endl;

    // Evaluate f(x0) and its derivative using forward mode
    double fx = f->eval(x0);
    double dfx_forward = f->derivative(x0);


    // Backward mode setup
    // function f and g definitions: Backward mode 
    // constants
    auto five = make_unique<ConstantNode>(5.0);
    auto four = make_unique<ConstantNode>(4.0);
    auto three = make_unique<ConstantNode>(3.0);

    // functions
    auto x = make_unique<InputNode>(x0);
    auto x2 = make_unique<PowerNode>(x.get(), 2);
    auto x2_minus_5 = make_unique<SubtractionNode>(x2.get(), five.get());
    auto three_x = make_unique<ProductNode>(three.get(), x.get());
    auto four_minus_3x = make_unique<SubtractionNode>(four.get(), three_x.get());
    auto g_backward = make_unique<ProductNode>(x2_minus_5.get(), four_minus_3x.get());
    auto log_g_backward = make_unique<LogNode>(g_backward.get());
    auto x_minus_4 = make_unique<SubtractionNode>(x.get(), four.get());
    auto log_g_over_x_minus_4 = make_unique<DivisionNode>(log_g_backward.get(), x_minus_4.get());
    auto x3 = make_unique<PowerNode>(x.get(), 3);
    auto five_plus_x3 = make_unique<AdditionNode>(five.get(), x3.get());
    auto f_b = make_unique<SubtractionNode>(five_plus_x3.get(), log_g_over_x_minus_4.get());

    // Forward pass
    x->forward();
    x2->forward();
    five->forward();
    four->forward();
    three->forward();
    x2_minus_5->forward();
    three_x->forward();
    four_minus_3x->forward();
    g_backward->forward();
    log_g_backward->forward();
    x_minus_4->forward();
    log_g_over_x_minus_4->forward();
    x3->forward();
    five_plus_x3->forward();
    f_b->forward();

    // Backward pass (gradient propagation; seed = 1)
    //calulate the dervative of x at x0 in backward mode
    f_b->backward(1.0);

    double dfx_backward = x->grad;

    // outputs
    cout << x0 << "\t" << fx << "\t" << dfx_backward << "\t" << dfx_forward << endl;

    return 0;
}