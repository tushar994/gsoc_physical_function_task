## Description of the different functions

- `log_abs` - This was very straightforward. simply find the log of the absolute value of x

- `sum_cos_sin` - This was also very straightforward.

- `approx_cos_p1` - This was strwightforward too, we simply find the taylr series approximation of cos(x) +1, which is given by
    ```
    2 - x^2/2! + x^4/4! - x^6/6!
    ```

- `integral_exp` - This is easy. As the integral of **e^x** is **e^x + C**, we simply find the difference between e^upper and e^lower

- `normed_exp` - This was just e^x divided by the result of `integral_exp`. It is assumed that upper and lower are not equal.

- `cos_exp` - This one took some time. Many approches were tried, where i used functions like tf.where and tf.map_fn and tf.vectorized_map. After trying alot of things, I simply decided to find the dimensions of x and do two different things for the two different shapes that were being given as input.

- `approx_cos_p1_custom_grad` - This one was easy. We simply return the same thing that is returned by `approx_cos_p1`, but we also return a gradient, which is given by 
    ```
    -x/1! + x^3/3! - x^5/5!
    ```

## Description of the class functions

### CosFunc

`value` - We return `cos(gamma*x)` here. I decided on this based on a comment i saw in the code "**# should be ~ cos(2. * 3.)**".

`integral` - This was simple, the integral of cos(x) is `-sin(x)`, so we found the difference between `-sin(upper)` and `-sin(lower)`

### CosPDF

`normed_value` - We divide cos(x) by the difference between `-sin(upper)` and `-sin(lower)`. It is assumed that upper and lower are not equal.