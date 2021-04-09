"""Google Summer of Code zfit lineshape implementation exercises.

In the following, use Tensorflow (tf) to implement the functions below.

The exercises have an increasing difficulty.

For tutorials on the topic, have a look at the official TensorFlow Documentation and 
the tutorial here: https://github.com/zfit/zfit-tutorials

YOU DON'T NEED TO IMPLEMENT EVERYTHING IF YOU CAN'T! Tell in the script what did not work and what you tried or
just omit an exercise you can't do. Also, make sure to try to solve the most difficult ones and see where you get
stuck.

Hint: for debugging, you can uncomment the `tf.function`, but make sure to put it back on afterwards.
"""
import tensorflow as tf
import abc


# This is an example to give you an idea
@tf.function(autograph=False)
def log_abs(x):
    """EXAMPLE IMPLEMENTATION: Return the log of the absolute of x element-wise"""
    return tf.math.log(tf.math.abs(x))


@tf.function(autograph=False)
def sum_cos_sin(x, coeff_cos, coeff_sin):
    """Return the sum of the cos and sin of x element-wise, cos and sin scaled by coeff_cos and coeff_sin respectively."""
    return tf.math.add( tf.math.multiply(coeff_sin, tf.math.sin(x)) , tf.math.multiply(coeff_cos, tf.math.cos(x))  )
    # pass


@tf.function(autograph=False)
def approx_cos_p1(x):
    """Return the approximation of cos(x) + 1 using a taylor/series expansion up to order 7"""
    result = tf.constant(2,dtype=tf.float64)
    fact = tf.constant(1,dtype=tf.float64)
    for i in range(2,7,2):
        fact = tf.math.multiply(fact,i-1)
        fact = tf.math.multiply(fact,i)
        if(i%4==2):
            result = tf.math.subtract(result, tf.math.divide(tf.math.pow(x,i),fact))
        else:
            result = tf.math.add(result, tf.math.divide(tf.math.pow(x,i),fact))
    return result



@tf.function(autograph=False)
def integral_exp(lower, upper):
    """Integrate the exponential function from lower to upper."""
    # We find e^upper - e^lower as the integral of e^x is e^x + C
    return tf.math.subtract(tf.math.exp(upper),tf.math.exp(lower))


@tf.function(autograph=False)
def normed_exp(x, lower, upper):
    """Calculate the normalized exp function `exp(x) / integral exp from lower to upper`"""
    return  tf.math.divide(tf.math.exp(x),integral_exp(lower,upper))

@tf.function(autograph=False)
def co_exp_single(x):
    return tf.cond(tf.math.greater(x, tf.constant(3.,dtype=tf.float64)), lambda : tf.math.cos(x), lambda : tf.cond(tf.math.less(x,tf.constant(-1., dtype=tf.float64)), lambda : tf.math.sin(x) + tf.math.cos(x),lambda : tf.math.exp(x)))

@tf.function(autograph=False)
def co_exp_three(x):
    return tf.map_fn(co_exp_single, x)

@tf.function(autograph=False)
def cos_exp(x):
    """Return the elementwise value of the function 'sum_cos_sin(x) for x < - 1; exp(x) for -1 < x < 3; cos(x) for x > 3."""
    if x.get_shape().as_list()!=[]:
        return tf.map_fn(co_exp_three, x)
    else:
        return tf.cond(tf.math.greater(x, tf.constant(3.,dtype=tf.float64)), lambda : tf.math.cos(x), lambda : tf.cond(tf.math.less(x,tf.constant(-1., dtype=tf.float64)), lambda : tf.math.sin(x) + tf.math.cos(x),lambda : tf.math.exp(x))) 



# use @tf.custom_gradient to add a custom gradient using again a series approximation
@tf.custom_gradient
# @tf.function(autograph=False)

def approx_cos_p1_custom_grad(x):
    """Return the approximation of cos(x) + 1 using a taylor/series expansion up to order 7 **with a custom gradient**
    """
    result = tf.constant(2,dtype=tf.float64)
    fact = tf.constant(1,dtype=tf.float64)
    for i in range(2,7,2):
        fact = tf.math.multiply(fact,i-1)
        fact = tf.math.multiply(fact,i)
        if(i%4==2):
            result = tf.math.subtract(result, tf.math.divide(tf.math.pow(x,i),fact))
        else:
            result = tf.math.add(result, tf.math.divide(tf.math.pow(x,i),fact))
    def grad(dy):
        result2 = tf.constant(0,dtype=tf.float64)
        fact2 = tf.constant(1,dtype=tf.float64)
        for i in range(1,6,2):
            if(i>1):
                fact2 = tf.math.multiply(fact2,i-1)
                fact2 = tf.math.multiply(fact2,i)
            if(i%4==1):
                result2 = tf.math.subtract(result2, tf.math.divide(tf.math.pow(x,i),fact2))
            else:
                result2 = tf.math.add(result2, tf.math.divide(tf.math.pow(x,i),fact2))
        return dy*result2
    return result, grad


# Classes
# This are less important than the TF exercise. If you know classes, they should be rather simple though.
# 1. make a class CosFunc that inherits from Func and takes one parameter omega. Implement the `value` and `integral` 
# methods (using tf)

class Func(abc.ABC):
    @abc.abstractmethod
    def value(self, x):
        pass

    @abc.abstractmethod
    def integral(self, lower, upper):
        pass

class CosFunc(Func):
    def __init__(self, omega):
        self.omega = omega

    def value(self,x):
        return tf.math.cos(self.omega * x)

    def integral(self, lower, upper):
        return tf.math.subtract(tf.math.sin(lower), tf.math.sin(upper))


# # create a class CosPDF, which inherits from PDF and takes two parameters to instantiate, `lower` and `upper`.
# # Then implement the `normed_value` which is the normalized value: value / integral
class PDF(Func):
    @abc.abstractmethod
    def normed_value(self, x):
        pass

class CosPDF(PDF):
    def __init__(self,lower,upper):
        self.upper = upper
        self.lower = lower

    def normed_value(self,x):
        return tf.math.divide( tf.math.cos(x), tf.math.subtract(tf.math.sin(self.lower), tf.math.sin(self.upper)) )
    
    def value(self, x):
        pass

    def integral(self, lower, upper):
        pass



if __name__ == '__main__':
    # test the functions here
    # it should work for both `x`
    use_single_val = True
    # use_single_val = False
    if use_single_val:
        x = tf.constant(1., dtype=tf.float64)
        # x = tf.constant(3.14, dtype=tf.float64)
        lower = tf.constant(0.3, dtype=tf.float64)
        # lower = tf.constant(-5.48522803, dtype=tf.float64)
        upper = tf.constant(5., dtype=tf.float64)
        coeff_sin = tf.constant(3.1, dtype=tf.float64)
        coeff_cos = tf.constant(2.1, dtype=tf.float64)
    else:
        shape = (10, 3)
        x = tf.random.uniform(shape=shape, minval=-4.5, maxval=2.3,dtype=tf.float64)
        lower = tf.random.uniform(shape=shape, minval=-5.5, maxval=-5.3,dtype=tf.float64)
        upper = tf.random.uniform(shape=shape, minval=4.1, maxval=5.2,dtype=tf.float64)
        coeff_sin = tf.random.uniform(shape=shape, minval=3.1, maxval=5.2,dtype=tf.float64)
        coeff_cos = tf.random.uniform(shape=shape, minval=1.2, maxval=2.23,dtype=tf.float64)
    # print(approx_cos_p1(x))
    print(sum_cos_sin(x,coeff_cos,coeff_sin))

    # get the gradient
    with tf.GradientTape() as tape:
        tape.watch(x)
        # your function here
        y = approx_cos_p1_custom_grad(x)
        # y = tf.math.cos(x)
    grad = tape.gradient(y, x)
    # print(x)
    print(grad)
    # print(tf.math.sin(x))
    # test the class here, uncomment below
    cos_func = CosFunc(omega=tf.constant(2., dtype=tf.float64))
    # cos_val = cos_func.value(tf.constant(3.))  # should be ~ cos(2. * 3.)
    cos_val = cos_func.value(x)
    cos_integral = cos_func.integral(lower,upper)
    print("cos functions")
    print(cos_val)
    print(cos_integral)
    print("cos second class")
    cos_pdf = CosPDF(lower = lower, upper = upper)  
    print(cos_pdf.normed_value(x))
