#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <boost/thread.hpp>

template < typename T >
T sigmoid1(T x)
{
    return 1.0f / (1.0f + exp(-x));
}

template < typename T >
T dsigmoid1(T x)
{
    return (1.0f - x)*x;
}

template < typename T >
T sigmoid2(T x)
{
    return log(1+exp(0.25*x));
}

template < typename T >
T dsigmoid2(T x)
{
    return 0.25/(1+exp(-0.25*x));
}

template < typename T >
T sigmoid(T x,int type)
{
    switch(type)
    {
        case 0:
            return sigmoid1(x);
        case 1:
            return sigmoid1(x);
    }
}

template < typename T >
T dsigmoid(T x,int type)
{
    switch(type)
    {
        case 0:
            return dsigmoid1(x);
        case 1:
            return dsigmoid1(x);
    }
}

template < typename T >
T max(T a,T b)
{
    return (a>b)?a:b;
}

template < typename T >
void apply_worker(std::vector<long> const & indices,long size,T * y,T * W,T * x)
{
  for(long k=0;k<indices.size();k++)
  {
    long i = indices[k];
    y[i] = 0;
    for(long j=0;j<size;j++)
    {
      y[i] += W[i*size+j]*x[j];
    }
  }
}

template < typename T >
void outer_product_worker(std::vector<long> const & indices,long size,T * H,T * A,T * B,T fact)
{
  for(long k=0;k<indices.size();k++)
  {
    long i = indices[k];
    for(long j=0;j<size;j++)
    {
      H[i*size+j] += A[i] * B[j] * fact;
    }
  }
}

template<typename T>
struct quasi_newton_info
{
    quasi_newton_info()
    {
        quasi_newton_update = false;
    }

    long get_size()
    {
        long size = 0;
        for(long layer = 0;layer < n_layers;layer++)
        {
            size += n_nodes[layer+1]*n_nodes[layer] + n_nodes[layer+1];
        }
        return size;
    }

    void init_gradient ()
    {
        long size = get_size();
        for(long layer = 0,k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    grad_tmp[k] = 0;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                grad_tmp[k] = 0;
            }
        }
    }

    void copy (T * src,T * dst,long size)
    {
        for(long k=0;k<size;k++)
        {
            dst[k] = src[k];
        }
    }

    void copy_avg (T * src,T * dst,T alph,long size)
    {
        for(long k=0;k<size;k++)
        {
            dst[k] += (src[k]-dst[k])*alph;
        }
    }

    bool quasi_newton_update;
    long n_layers;
    T *** weights_neuron;
    T **  weights_bias;
    std::vector<long> n_nodes;
    T * grad_tmp;
    T * grad_1;
    T * grad_2;
    T * Y;
    T * dX;
    T * B;
    T * H;
    T alpha;

    void init_QuasiNewton()
    {
        long size = get_size();
        grad_tmp = new T[size];
        init_gradient();
        grad_1 = new T[size];
        grad_2 = new T[size];
        copy(grad_tmp,grad_1,size);
        copy(grad_tmp,grad_2,size);
        B = new T[size*size];
        T * B_tmp = init_B();
        copy(B_tmp,B,size*size);
        delete [] B_tmp;
        H = new T[size*size];
        T * H_tmp = init_H();
        copy(H_tmp,H,size*size);
        delete [] H_tmp;
        dX = new T[size*size];
        T * dX_tmp = get_dx();
        copy(dX_tmp,dX,size);
        delete [] dX_tmp;
        Y = new T[size*size];
    }

    T * init_B()
    {
        long size = get_size();
        T * B = new T[size*size];
        for(long t=0;t<size*size;t++)
        {
            B[t] = 0;
        }
        for(long layer = 0, k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    B[k*size+k] = weights_neuron[layer][i][j];
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                B[k*size+k] = weights_bias[layer][i];
            }
        }
        return B;
    }

    T * init_H()
    {
        long size = get_size();
        T * H = new T[size*size];
        for(long t=0;t<size*size;t++)
        {
            H[t] = 0;
        }
        for(long layer = 0, k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    H[k*size+k] = -1;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                H[k*size+k] = -1;
            }
        }
        return H;
    }

    void update_QuasiNewton()
    {
        long size = get_size();
        copy_avg(grad_2,grad_1,0.1,size);
        copy(grad_tmp,grad_2,size);
        T * Y_tmp = get_y();
        copy(Y_tmp,Y,size);
        delete [] Y_tmp;
        T * dX_tmp = get_dx();
        copy(dX_tmp,dX,size);
        delete [] dX_tmp;
    }

    T * get_y ()
    {
        long size = get_size();
        T * y = new T[size];
        //T y_m = 0;
        for(long k=0;k<size;k++)
        {
            y[k] = grad_2[k] - grad_1[k];
            //y_m = max(y_m,fabs(y[k]));
        }
        return y;
    }

    T * get_dx ()
    {
        long size = get_size();
        T * dx = apply(H,grad_1);
        for(long k=0;k<size;k++)
        {
            dx[k] *= -alpha;
        }
        return dx;
    }

    T * get_outer_product(T * a,T * b)
    {
        long size = get_size();
        long prod_size = size*size;
        T * prod = new T[prod_size];
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            prod[k] = a[i]*b[j];
          }
        }
        return prod;
    }

    T get_inner_product(T * a,T * b)
    {
        T ret = 0;
        long size = get_size();
        for(long i=0;i<size;i++)
        {
            ret += a[i]*b[i];
        }
        T eps = 1e-3;
        if(ret<0)
        {
            ret -= eps;
        }
        else
        {
            ret += eps;
        }
        return ret;
    }

    T * apply(T * W, T * x)
    {
        long size = get_size();
        T * y = new T[size];
        std::vector<boost::thread * > threads;
        long num_cpu = boost::thread::hardware_concurrency();
        std::vector<std::vector<long> > indices(num_cpu);
        for(long i=0;i<size;i++)
        {
          indices[i%num_cpu].push_back(i);
        }
        for(long i=0;i<num_cpu;i++)
        {
          threads.push_back(new boost::thread(apply_worker<T>,indices[i],size,&y[0],&W[0],&x[0]));
        }
        for(long i=0;i<threads.size();i++)
        {
          threads[i]->join();
          delete threads[i];
        }
        return y;
    }

    T * apply_t(T * x, T * W)
    {
        long size = get_size();
        T * y = new T[size];
        for(long i=0,k=0;i<size;i++)
        {
          y[i] = 0;
          for(long j=0;j<size;j++,k++)
          {
            y[i] += W[size*j+i]*x[j];
          }
        }
        return y;
    }

    T limit(T x,T eps)
    {
        if(x>0)
        {
            if(x>eps)return eps;
        }
        else
        {
            if(x<-eps)return -eps;
        }
        return x;
    }

    // SR1
    void SR1_update()
    {
        long size = get_size();
        T * dx_Hy = apply(H,Y);
        for(long i=0;i<size;i++)
        {
          dx_Hy[i] = dX[i] - dx_Hy[i];
        }
        T inner = 1.0 / (get_inner_product(dx_Hy,Y));
        std::vector<boost::thread * > threads;
        long num_cpu = boost::thread::hardware_concurrency();
        std::vector<std::vector<long> > indices(num_cpu);
        for(long i=0;i<size;i++)
        {
          indices[i%num_cpu].push_back(i);
        }
        for(long i=0;i<num_cpu;i++)
        {
          threads.push_back(new boost::thread(outer_product_worker<T>,indices[i],size,&H[0],&dx_Hy[0],&dx_Hy[0],inner));
        }
        for(long i=0;i<threads.size();i++)
        {
          threads[i]->join();
          delete threads[i];
        }
        delete [] dx_Hy;
    }

    // Broyden
    void Broyden_update()
    {
        long size = get_size();
        T * dx_Hy = apply(H,Y);
        for(long i=0;i<size;i++)
        {
          dx_Hy[i] = dX[i] - dx_Hy[i];
        }
        T * xH = apply_t(dX,H);
        T * outer = get_outer_product(dx_Hy,xH);
        T inner = 1.0 / (get_inner_product(xH,Y));
        for(long i=0;i<size*size;i++)
        {
          H[i] += outer[i] * inner;
        }
        delete [] dx_Hy;
        delete [] xH;
        delete [] outer;
    }

    // DFP
    void DFP_update()
    {
        long size = get_size();
        T * Hy = apply(H,Y);
        T * outer_2 = get_outer_product(Hy,Hy);
        T inner_2 = -1.0 / (get_inner_product(Hy,Y));
        T * outer_1 = get_outer_product(dX,dX);
        T inner_1 = 1.0 / (get_inner_product(dX,Y));
        for(long i=0;i<size*size;i++)
        {
          H[i] += outer_1[i] * inner_1 + outer_2[i] * inner_2;
        }
        delete [] outer_2;
        delete [] outer_1;
        delete [] Hy;
    }

    T * apply_M(T * A, T * B)
    {
        long size = get_size();
        T * C = new T[size*size];
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            C[k] = 0;
            for(long t=0;t<size;t++)
            {
              C[k] += A[i*size+t]*B[t*size+j];
            }
          }
        }
        return C;
    }

    // BFGS
    void BFGS_update()
    {
        long size = get_size();
        T inner = 1.0 / (get_inner_product(Y,dX));
        T * outer_xx = get_outer_product(dX,dX);
        T * outer_xy = get_outer_product(dX,Y);
        T * outer_yx = get_outer_product(Y,dX);
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            if(i==j)
            {
              outer_xy[k] = 1-outer_xy[k]*inner;
              outer_yx[k] = 1-outer_yx[k]*inner;
            }
            else
            {
              outer_xy[k] = -outer_xy[k]*inner;
              outer_yx[k] = -outer_yx[k]*inner;
            }
            outer_xx[k] = outer_xx[k]*inner;
          }
        }
        T * F = apply_M(outer_xy,H);
        T * G = apply_M(F,outer_yx);
        for(long i=0;i<size*size;i++)
        {
          H[i] = G[i] + outer_xx[i];
        }
        delete [] F;
        delete [] G;
        delete [] outer_xx;
        delete [] outer_xy;
        delete [] outer_yx;
    }

};

template<typename T>
struct training_info
{

    quasi_newton_info<T> * quasi_newton;

    std::vector<long> n_nodes;
    T **  activation_values;
    T **  deltas;
    long n_variables;
    long n_labels;
    long n_layers;
    long n_elements;

    T *** weights_neuron;
    T **  weights_bias;
    T *** partial_weights_neuron;
    T **  partial_weights_bias;

    T partial_error;

    T epsilon;

    int type;

    training_info()
    {

    }

    void init(T _alpha)
    {
        type = 0;
        partial_error = 0;
        activation_values  = new T*[n_nodes.size()];
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            activation_values [layer] = new T[n_nodes[layer]];
        }
        deltas = new T*[n_nodes.size()];
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            deltas[layer] = new T[n_nodes[layer]];
        }
        partial_weights_neuron = new T**[n_layers];
        partial_weights_bias = new T*[n_layers];
        for(long layer = 0;layer < n_layers;layer++)
        {
            partial_weights_neuron[layer] = new T*[n_nodes[layer+1]];
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                partial_weights_neuron[layer][i] = new T[n_nodes[layer]];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    partial_weights_neuron[layer][i][j] = 0;
                }
            }
            partial_weights_bias[layer] = new T[n_nodes[layer+1]];
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                partial_weights_bias[layer][i] = 0;
            }
        }
    }

    void destroy()
    {
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            delete [] activation_values [layer];
        }
        delete [] activation_values;
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            delete [] deltas [layer];
        }
        delete [] deltas;
        for(long layer = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                delete [] partial_weights_neuron[layer][i];
            }
            delete [] partial_weights_neuron[layer];
        }
        delete [] partial_weights_neuron;
        for(long layer = 0;layer < n_layers;layer++)
        {
            delete [] partial_weights_bias[layer];
        }
        delete [] partial_weights_bias;
    }

    void update_gradient ()
    {
        for(long layer = 0,k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    quasi_newton->grad_tmp[k] += partial_weights_neuron[layer][i][j] / n_elements;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                quasi_newton->grad_tmp[k] += partial_weights_bias[layer][i] / n_elements;
            }
        }
    }

    void globalUpdate()
    {
        if(quasi_newton->quasi_newton_update)
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                for(long i=0;i<n_nodes[layer+1];i++)
                {
                    for(long j=0;j<n_nodes[layer];j++,k++)
                    {
                        weights_neuron[layer][i][j] += quasi_newton->dX[k];
                    }
                }
                for(long i=0;i<n_nodes[layer+1];i++,k++)
                {
                    weights_bias[layer][i] += quasi_newton->dX[k];
                }
            }
        }
        else
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                for(long i=0;i<n_nodes[layer+1];i++)
                {
                    for(long j=0;j<n_nodes[layer];j++,k++)
                    {
                        weights_neuron[layer][i][j] += epsilon * quasi_newton->grad_tmp[k];
                    }
                }
                for(long i=0;i<n_nodes[layer+1];i++,k++)
                {
                    weights_bias[layer][i] += epsilon * quasi_newton->grad_tmp[k];
                }
            }
        }
    }

};

template<typename T>
void training_worker(training_info<T> * g,std::vector<long> const & vrtx,T * variables,T * labels)
{
    
    for(long n=0;n<vrtx.size();n++)
    {

        // initialize input activations
        for(long i=0;i<g->n_nodes[0];i++)
        {
            g->activation_values[0][i] = variables[vrtx[n]*g->n_variables+i];
        }
        // forward propagation
        for(long layer = 0; layer < g->n_layers; layer++)
        {
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                T sum = g->weights_bias[layer][i];
                for(long j=0;j<g->n_nodes[layer];j++)
                {
                    sum += g->activation_values[layer][j] * g->weights_neuron[layer][i][j];
                }
                g->activation_values[layer+1][i] = sigmoid(sum,g->type);
                //std::cout << g->activation_values[layer+1][i] << '\t';
            }
            //std::cout << std::endl;
        }
        long last_layer = g->n_nodes.size()-2;
        // initialize observed labels
        for(long i=0;i<g->n_nodes[last_layer];i++)
        {
            g->deltas[last_layer+1][i] = labels[vrtx[n]*g->n_labels+i] - g->activation_values[last_layer][i];
            g->partial_error += fabs(g->deltas[last_layer+1][i]);
            //std::cout << g->deltas[last_layer+1][i] << '\t';
        }
        //std::cout << std::endl;
        // back propagation
        for(long layer = g->n_layers-1; layer >= 0; layer--)
        {
            // back propagate deltas
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                g->deltas[layer+1][i] = 0;
                for(long j=0;j<g->n_nodes[layer+2];j++)
                {
                    if(layer+1==last_layer)
                    {
                        g->deltas[layer+1][i] += dsigmoid(g->activation_values[layer+1][i],g->type)*g->deltas[layer+2][j];
                    }
                    else
                    {
                        g->deltas[layer+1][i] += dsigmoid(g->activation_values[layer+1][i],g->type)*g->deltas[layer+2][j]*g->weights_neuron[layer+1][j][i];
                    }
                }
                //std::cout << g->deltas[layer+1][i] << '\t';
            }
            //std::cout << std::endl;
            //std::cout << "biases" << std::endl;
            // biases
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                g->partial_weights_bias[layer][i] += g->deltas[layer+1][i];
                //std::cout << g->partial_weights_bias[layer][i] << '\t';
            }
            //std::cout << std::endl;
            //std::cout << "neuron weights" << std::endl;
            // neuron weights
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                for(long j=0;j<g->n_nodes[layer];j++)
                {
                    g->partial_weights_neuron[layer][i][j] += g->activation_values[layer][j] * g->deltas[layer+1][i];
                    //std::cout << g->partial_weights_neuron[layer][i][j] << '\t';
                }
                //std::cout << std::endl;
            }
            //std::cout << std::endl;
        }
        //char ch;
        //std::cin >> ch;
    }

}

template<typename T>
struct Perceptron
{
    quasi_newton_info<T> * quasi_newton;

    T ierror;
    T perror;

    T *** weights_neuron;
    T **  weights_bias;
    T **  activation_values;
    T **  activation_values1;
    T **  deltas;

    long n_inputs;
    long n_outputs;
    long n_layers;
    std::vector<long> n_nodes;

    T epsilon;
    T alpha;
    int sigmoid_type;

    // std::vector<long> nodes;
    // nodes.push_back(2); // inputs
    // nodes.push_back(3); // hidden layer
    // nodes.push_back(1); // output layer
    // nodes.push_back(1); // outputs
    Perceptron(std::vector<long> p_nodes)
    {

        quasi_newton = NULL;

        sigmoid_type = 0;
        alpha = 0.1;

        ierror = 1e10;
        perror = 1e10;

        n_nodes = p_nodes;
        n_inputs = n_nodes[0];
        n_outputs = n_nodes[n_nodes.size()-1];
        n_layers = n_nodes.size()-2; // first and last numbers and output and input dimensions, so we have n-2 layers

        weights_neuron = new T**[n_layers];
        weights_bias = new T*[n_layers];
        activation_values  = new T*[n_nodes.size()];
        activation_values1 = new T*[n_nodes.size()];
        deltas = new T*[n_nodes.size()];
        
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            activation_values [layer] = new T[n_nodes[layer]];
            activation_values1[layer] = new T[n_nodes[layer]];
            deltas[layer] = new T[n_nodes[layer]];
        }

        for(long layer = 0;layer < n_layers;layer++)
        {
            weights_neuron[layer] = new T*[n_nodes[layer+1]];
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                weights_neuron[layer][i] = new T[n_nodes[layer]];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    weights_neuron[layer][i][j] = -1.0 + 2.0 * ((rand()%10000)/10000.0);
                }
            }
            weights_bias[layer] = new T[n_nodes[layer+1]];
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                weights_bias[layer][i] = -1.0 + 2.0 * ((rand()%10000)/10000.0);
            }
        }

        //weights_neuron[0][0][0] = .1;        weights_neuron[0][0][1] = .2;
        //weights_neuron[0][1][0] = .3;        weights_neuron[0][1][1] = .4;
        //weights_neuron[0][2][0] = .5;        weights_neuron[0][2][1] = .6;

        //weights_bias[0][0] = .1;
        //weights_bias[0][1] = .2;
        //weights_bias[0][2] = .3;

        //weights_neuron[1][0][0] = .6;        weights_neuron[1][0][1] = .7;      weights_neuron[1][0][2] = .8;

        //weights_bias[1][0] = .5;

    }

    T * model(long n_elements,long n_labels,T * variables)
    {
        T * labels = new T[n_labels];
        // initialize input activations
        for(long i=0;i<n_nodes[0];i++)
        {
            activation_values1[0][i] = variables[i];
        }
        // forward propagation
        for(long layer = 0; layer < n_layers; layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                T sum = weights_bias[layer][i];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    sum += activation_values1[layer][j] * weights_neuron[layer][i][j];
                }
                activation_values1[layer+1][i] = sigmoid(sum,get_sigmoid());
            }
        }
        long last_layer = n_nodes.size()-2;
        for(long i=0;i<n_labels;i++)
        {
            labels[i] = activation_values1[last_layer][i];
        }
        return labels;
    }

    int get_sigmoid()
    {
        return sigmoid_type;
    }

    void train(int p_sigmoid_type,T p_epsilon,long n_iterations,long n_elements,long n_variables,T * variables,long n_labels, T * labels)
    {
        sigmoid_type = p_sigmoid_type;
        epsilon = p_epsilon;
        if(n_variables != n_nodes[0]){std::cout << "error 789437248932748293" << std::endl;exit(0);}
        quasi_newton = new quasi_newton_info<T>();
        quasi_newton->alpha = alpha;
        quasi_newton->n_nodes = n_nodes;
        quasi_newton->n_layers = n_layers;
        quasi_newton->weights_neuron = weights_neuron;
        quasi_newton->weights_bias = weights_bias;
        quasi_newton->init_QuasiNewton();
        ierror = 1e10;
        bool init = true;
        perror = 1e10;
        for(long iter = 0; iter < n_iterations; iter++)
        {
            T error = 0;


            //////////////////////////////////////////////////////////////////////////////////
            //                                                                              //
            //          Multi-threaded block                                                //
            //                                                                              //
            //////////////////////////////////////////////////////////////////////////////////
            std::vector<boost::thread*> threads;
            std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
            std::vector<training_info<T>*> g;
            for(long i=0;i<n_elements;i++)
            {
              vrtx[i%vrtx.size()].push_back(i);
            }
            for(long i=0;i<vrtx.size();i++)
            {
              g.push_back(new training_info<T>());
            }
            quasi_newton->init_gradient();
            for(long thread=0;thread<vrtx.size();thread++)
            {
              g[thread]->quasi_newton = quasi_newton;
              g[thread]->n_nodes = n_nodes;
              g[thread]->n_elements = n_elements;
              g[thread]->n_variables = n_variables;
              g[thread]->n_labels = n_labels;
              g[thread]->n_layers = n_layers;
              g[thread]->weights_neuron = weights_neuron;
              g[thread]->weights_bias = weights_bias;
              g[thread]->epsilon = epsilon;
              g[thread]->type = get_sigmoid();

              g[thread]->init(alpha);
              threads.push_back(new boost::thread(training_worker<T>,g[thread],vrtx[thread],variables,labels));
            }
            for(long thread=0;thread<vrtx.size();thread++)
            {
              threads[thread]->join();
              g[thread]->update_gradient();
              delete threads[thread];
            }
            quasi_newton->update_QuasiNewton();
            quasi_newton->SR1_update();
            for(long thread=0;thread<vrtx.size();thread++)
            {
              g[thread]->globalUpdate();
              error += g[thread]->partial_error;
              g[thread]->destroy();
              delete g[thread];
            }
            threads.clear();
            vrtx.clear();
            g.clear();

            static int cnt1 = 0;
            if(cnt1%10==0)
            std::cout << "quasi_newton_update=" << quasi_newton->quasi_newton_update << "\ttype=" << sigmoid_type << "\tepsilon=" << epsilon << "\talpha=" << alpha << '\t' << "error=" << error << "\tdiff=" << (error-perror) << std::endl;
            cnt1++;
            perror = error;
            if(init)
            {
                ierror = error;
                init = false;
            }

            //char ch;
            //std::cin >> ch;

        }
    }

};

double *  in_dat = NULL;
double * out_dat = NULL;

Perceptron<double> * perceptron = NULL;

int nx = 7, ny = 7;
void
drawStuff(void)
{
  if(perceptron != NULL)
  {
    int num_inputs = nx+ny;
    double * in = new double[nx+ny];
    double dx = 2/200.0f;
    double dy = 2/200.0f;
    glBegin(GL_QUADS);
    for (double x = -1; x <= 1; x+=dx) 
    for (double y = -1; y <= 1; y+=dy) 
    {
      int bX = pow(2,nx)*(x+1)/2;
      int bY = pow(2,ny)*(y+1)/2;
      for(long i=0;i<nx;i++)
      {
        in[i] = bX%2==0;
        bX/=2;
      }
      for(long i=nx;i<nx+ny;i++)
      {
        in[i] = bY%2==0;
        bY/=2;
      }
      double * out = perceptron->model(2,1,in);
      glColor3f(out[0],out[0],out[0]);
      delete [] out;
      out = NULL;
      glVertex3f(x   ,y   ,0);
      glVertex3f(x+dx,y   ,0);
      glVertex3f(x+dx,y+dy,0);
      glVertex3f(x   ,y+dy,0);
    }
    glEnd();
    delete [] in;
    in = NULL;
  }
}

void
display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawStuff();
  glutSwapBuffers();
}

void 
idle(void)
{
  usleep(10000);
  glutPostRedisplay();
}

void
init(void)
{

  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 1.8,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */

  /* Adjust cube position to be aesthetic angle. */
  glTranslatef(0.0, 0.0, -1.0);
  glRotatef(0, 1.0, 0.0, 0.0);
  glRotatef(0, 0.0, 0.0, 1.0);
  glEnable (GL_BLEND); 
  //glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE);
  glBlendEquation(GL_FUNC_ADD);

}

void keyboard(unsigned char key,int x,int y)
{
  switch(key)
  {
    case 27:exit(1);break;
    case 'w':perceptron->epsilon *= 1.1; break;
    case 's':perceptron->epsilon /= 1.1; break;
    case 'r':perceptron->alpha *= 1.1;perceptron->quasi_newton->alpha = perceptron->alpha; break;
    case 'f':perceptron->alpha /= 1.1;perceptron->quasi_newton->alpha = perceptron->alpha; break;
    case 'a':perceptron->sigmoid_type = (perceptron->sigmoid_type+1)%2; break;
    case 'd':perceptron->sigmoid_type = (perceptron->sigmoid_type+1)%2; break;
    case 'z':if(perceptron->quasi_newton!=NULL){perceptron->quasi_newton->quasi_newton_update=!perceptron->quasi_newton->quasi_newton_update;}break;
    default:break;
  }
}

double norm(double * dat,long size)
{
  double ret = 0;
  for(long i=0;i<size;i++)
  {
    ret += dat[i]*dat[i];
  }
  return sqrt(ret);
}

void zero(double * dat,long size)
{
  for(long i=0;i<size;i++)
  {
    dat[i] = 0;
  }
}

void constant(double * dat,double val,long size)
{
  for(long i=0;i<size;i++)
  {
    dat[i] = (-1+2*((rand()%10000)/10000.0f))*val;
  }
}

void add(double * A, double * dA, double epsilon, long size)
{
  for(long i=0;i<size;i++)
  {
    A[i] += epsilon * dA[i];
  }
}

struct gradient_info
{
  long n;
  long v;
  long h;
  double * vis0;
  double * hid0;
  double * vis;
  double * hid;
  double * dW;
  double * dc;
  double * db;
  double partial_err;
  double * partial_dW;
  double * partial_dc;
  double * partial_db;
  void init()
  {
    partial_err = 0;
    partial_dW = new double[h*v];
    for(int i=0;i<h*v;i++)partial_dW[i]=0;
    partial_dc = new double[h];
    for(int i=0;i<h;i++)partial_dc[i]=0;
    partial_db = new double[v];
    for(int i=0;i<v;i++)partial_db[i]=0;
  }
  void destroy()
  {
    delete [] partial_dW;
    delete [] partial_dc;
    delete [] partial_db;
  }
  void globalUpdate()
  {
    for(int i=0;i<h*v;i++)
        dW[i] += partial_dW[i];
    for(int i=0;i<h;i++)
        dc[i] += partial_dc[i];
    for(int i=0;i<v;i++)
        db[i] += partial_db[i];
  }
};

void gradient_worker(gradient_info * g,std::vector<long> const & vrtx)
{
  double factor = 1.0f / g->n;
  double factorv= 1.0f / (g->v*g->v);
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long i=0;i<g->v;i++)
    {
      for(long j=0;j<g->h;j++)
      {
        g->partial_dW[i*g->h+j] -= factor * (g->vis0[k*g->v+i]*g->hid0[k*g->h+j] - g->vis[k*g->v+i]*g->hid[k*g->h+j]);
      }
    }

    for(long j=0;j<g->h;j++)
    {
      g->partial_dc[j] -= factor * (g->hid0[k*g->h+j]*g->hid0[k*g->h+j] - g->hid[k*g->h+j]*g->hid[k*g->h+j]);
    }

    for(long i=0;i<g->v;i++)
    {
      g->partial_db[i] -= factor * (g->vis0[k*g->v+i]*g->vis0[k*g->v+i] - g->vis[k*g->v+i]*g->vis[k*g->v+i]);
    }

    for(long i=0;i<g->v;i++)
    {
      g->partial_err += factorv * (g->vis0[k*g->v+i]-g->vis[k*g->v+i])*(g->vis0[k*g->v+i]-g->vis[k*g->v+i]);
    }
  }
}

void vis2hid_worker(const double * X,double * H,long h,long v,double * c,double * W,std::vector<long> const & vrtx)
{
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long j=0;j<h;j++)
    {
      H[k*h+j] = c[j]; 
      for(long i=0;i<v;i++)
      {
        H[k*h+j] += W[i*h+j] * X[k*v+i];
      }
      H[k*h+j] = 1.0f/(1.0f + exp(-H[k*h+j]));
    }
  }
}

void hid2vis_worker(const double * H,double * V,long h,long v,double * b,double * W,std::vector<long> const & vrtx)
{
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long i=0;i<v;i++)
    {
      V[k*v+i] = b[i]; 
      for(long j=0;j<h;j++)
      {
        V[k*v+i] += W[i*h+j] * H[k*h+j];
      }
      V[k*v+i] = 1.0f/(1.0f + exp(-V[k*v+i]));
    }
  }
}

struct RBM
{
  long h; // number hidden elements
  long v; // number visible elements
  long n; // number of samples
  double * c; // bias term for hidden state, R^h
  double * b; // bias term for visible state, R^v
  double * W; // weight matrix R^h*v
  double * X; // input data, binary [0,1], v*n

  double * vis0;
  double * hid0;
  double * vis;
  double * hid;
  double * dW;
  double * dc;
  double * db;

  RBM(long _v,long _h,double * _W,double * _b,double * _c,long _n,double * _X)
  {
    //for(long k=0;k<100;k++)
    //  std::cout << _X[k] << "\t";
    //std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = _c;
    b = _b;
    W = _W;

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }
  RBM(long _v,long _h,long _n,double* _X)
  {
    //for(long k=0;k<100;k++)
    //  std::cout << _X[k] << "\t";
    //std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = new double[h];
    b = new double[v];
    W = new double[h*v];
    constant(c,0.5f,h);
    constant(b,0.5f,v);
    constant(W,0.5f,v*h);

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }

  void init(int offset)
  {
    boost::posix_time::ptime time_start(boost::posix_time::microsec_clock::local_time());
    if(vis0==NULL)vis0 = new double[n*v];
    if(hid0==NULL)hid0 = new double[n*h];
    if(vis==NULL)vis = new double[n*v];
    if(hid==NULL)hid = new double[n*h];
    if(dW==NULL)dW = new double[h*v];
    if(dc==NULL)dc = new double[h];
    if(db==NULL)db = new double[v];

    //std::cout << "n*v=" << n*v << std::endl;
    //std::cout << "offset=" << offset << std::endl;
    for(long i=0,size=n*v;i<size;i++)
    {
      vis0[i] = X[i+offset];
    }

    vis2hid(vis0,hid0);
    boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration(time_end - time_start);
    //std::cout << "init timing:" << duration << '\n';
  }

  void cd(long nGS,double epsilon,int offset=0,bool bottleneck=false)
  {
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());
    //std::cout << "cd" << std::endl;

    // CD Contrastive divergence (Hlongon's CD(k))
    //   [dW, db, dc, act] = cd(self, X) returns the gradients of
    //   the weihgts, visible and hidden biases using Hlongon's
    //   approximated CD. The sum of the average hidden units
    //   activity is returned in act as well.

    for(long i=0;i<n*h;i++)
    {
      hid[i] = hid0[i];
    }
    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    //std::cout << "cd timing 1:" << duration10 << '\n';

    for (long iter = 1;iter<=nGS;iter++)
    {
      //std::cout << "iter=" << iter << std::endl;
      // sampling
      hid2vis(hid,vis);
      vis2hid(vis,hid);

// Preview stuff
#if 0
      long off = dat_offset%(n);
      long offv = off*v;
      long offh = off*h;
      long off_preview = off*(3*WIN*WIN+10);
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis_preview[k] = vis[offv+k];
          vis_previewG[k] = vis[offv+k+WIN*WIN];
          vis_previewB[k] = vis[offv+k+2*WIN*WIN];
        }
      }
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis1_preview[k] = orig_arr[offset+off_preview+k];
          vis1_previewG[k] = orig_arr[offset+off_preview+k+WIN*WIN];
          vis1_previewB[k] = orig_arr[offset+off_preview+k+2*WIN*WIN];
        }
      }
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis0_preview[k] = vis0[offv+k];
          vis0_previewG[k] = vis0[offv+k+WIN*WIN];
          vis0_previewB[k] = vis0[offv+k+2*WIN*WIN];
        }
      }
#endif

    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    //std::cout << "cd timing 2:" << duration21 << '\n';
  
    zero(dW,v*h);
    zero(dc,h);
    zero(db,v);
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    //std::cout << "cd timing 3:" << duration32 << '\n';
    double * err = new double(0);
    gradient_update(n,vis0,hid0,vis,hid,dW,dc,db,err);
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    //std::cout << "cd timing 4:" << duration43 << '\n';
    *err = sqrt(*err);
    //for(int t=2;t<3&&t<errs.size();t++)
    //  *err += (errs[errs.size()+1-t]-*err)/t;
    static int cnt2 = 0;
    if(cnt2%10==0)
    std::cout << "Boltzmann error:" << *err << std::endl;
    cnt2++;
    //errs.push_back(*err);
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    //std::cout << "cd timing 5:" << duration54 << '\n';
    //std::cout << "epsilon = " << epsilon << std::endl;
    add(W,dW,-epsilon,v*h);
    add(c,dc,-epsilon,h);
    add(b,db,-epsilon,v);

    //std::cout << "dW norm = " << norm(dW,v*h) << std::endl;
    //std::cout << "dc norm = " << norm(dc,h) << std::endl;
    //std::cout << "db norm = " << norm(db,v) << std::endl;
    //std::cout << "W norm = " << norm(W,v*h) << std::endl;
    //std::cout << "c norm = " << norm(c,h) << std::endl;
    //std::cout << "b norm = " << norm(b,v) << std::endl;
    //std::cout << "err = " << *err << std::endl;
    delete err;

    boost::posix_time::ptime time_6(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration65(time_6 - time_5);
    //std::cout << "cd timing 6:" << duration65 << '\n';
    //char ch;
    //std::cin >> ch;
  }

  void sigmoid(double * p,double * X,long n)
  {
    for(long i=0;i<n;i++)
    {
      p[i] = 1.0f/(1.0f + exp(-X[i]));
    }
  }

  void vis2hid_simple(const double * X,double * H)
  {
    {
      for(long j=0;j<h;j++)
      {
        H[j] = c[j]; 
        for(long i=0;i<v;i++)
        {
          H[j] += W[i*h+j] * X[i];
        }
        H[j] = 1.0f/(1.0f + exp(-H[j]));
      }
    }
  }

  void hid2vis_simple(const double * H,double * V)
  {
    {
      for(long i=0;i<v;i++)
      {
        V[i] = b[i]; 
        for(long j=0;j<h;j++)
        {
          V[i] += W[i*h+j] * H[j];
        }
        V[i] = 1.0f/(1.0f + exp(-V[i]));
      }
    }
  }

  void vis2hid(const double * X,double * H)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(vis2hid_worker,X,H,h,v,c,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
    threads.clear();
    vrtx.clear();
  }

  void gradient_update(long n,double * vis0,double * hid0,double * vis,double * hid,double * dW,double * dc,double * db,double * err)
  {
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());

    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    std::vector<gradient_info*> g;

    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    //std::cout << "gradient update timing 1:" << duration10 << '\n';

    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    //std::cout << "gradient update timing 2:" << duration21 << '\n';
    for(long i=0;i<vrtx.size();i++)
    {
      g.push_back(new gradient_info());
    }
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    //std::cout << "gradient update timing 3:" << duration32 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      g[thread]->n = n;
      g[thread]->v = v;
      g[thread]->h = h;
      g[thread]->vis0 = vis0;
      g[thread]->hid0 = hid0;
      g[thread]->vis = vis;
      g[thread]->hid = hid;
      g[thread]->dW = dW;
      g[thread]->dc = dc;
      g[thread]->db = db;
      g[thread]->init();
      threads.push_back(new boost::thread(gradient_worker,g[thread],vrtx[thread]));
    }
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    //std::cout << "gradient update timing 4:" << duration43 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
      g[thread]->globalUpdate();
      *err += g[thread]->partial_err;
      g[thread]->destroy();
      delete g[thread];
    }
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    //std::cout << "gradient update timing 5:" << duration54 << '\n';
    threads.clear();
    vrtx.clear();
    g.clear();
  }
  
  void hid2vis(const double * H,double * V)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(hid2vis_worker,H,V,h,v,b,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
    threads.clear();
    vrtx.clear();
  }

  void print()
  {
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "W:" << std::endl;
    for(int i=0;i<v;i++)
    {
      for(int j=0;j<h;j++)
      {
        std::cout << W[i*h+j] << '\t';
      }
      std::cout << '\n';
    }
    std::cout << "b:" << std::endl;
    for(int i=0;i<v;i++)
    {
      {
        std::cout << b[i] << '\t';
      }
      std::cout << '\n';
    }
    std::cout << "c:" << std::endl;
    {
      for(int j=0;j<h;j++)
      {
        std::cout << c[j] << '\t';
      }
      std::cout << '\n';
    }
  }

};

struct DataUnit
{
  DataUnit *   hidden;
  DataUnit *  visible;
  DataUnit * visible0;
  long h,v;
  double * W;
  double * b;
  double * c;
  RBM * rbm;
  long num_iters;
  long batch_iter;
  DataUnit(long _v,long _h,long _num_iters = 100,long _batch_iter = 1)
  {
    num_iters = _num_iters;
    batch_iter = _batch_iter;
    v = _v;
    h = _h;
    W = new double[v*h];
    b = new double[v];
    c = new double[h];
    constant(c,0.5f,h);
    constant(b,0.5f,v);
    constant(W,0.5f,v*h);
      hidden = NULL;
     visible = NULL;
    visible0 = NULL;
  }

  void train(double * dat, long n, long total_n,int n_cd,double epsilon,long n_var)
  {
    // RBM(long _v,long _h,double * _W,double * _b,double * _c,long _n,double * _X)
    rbm = new RBM(v,h,W,b,c,n,dat);
    for(long i=0;i<num_iters;i++)
    {
      //std::cout << "DataUnit::train i=" << i << std::endl;
      long offset = (rand()%(total_n-n));
      for(long k=0;k<batch_iter;k++)
      {
        rbm->init(offset);
        //std::cout << "prog:" << 100*(double)k/batch_iter << "%" << std::endl;
        rbm->cd(n_cd,epsilon,offset*n_var);
      }
    }
    //char ch;
    //std::cin >> ch;
  }

  void transform(double* X,double* Y)
  {
    rbm->vis2hid(X,Y);
  }

  void print()
  {
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "W:" << std::endl;
    for(int i=0;i<v;i++)
    {
      for(int j=0;j<h;j++)
      {
        std::cout << W[i*h+j] << '\t';
      }
      std::cout << '\n';
    }
    std::cout << "b:" << std::endl;
    for(int i=0;i<v;i++)
    {
      {
        std::cout << b[i] << '\t';
      }
      std::cout << '\n';
    }
    std::cout << "c:" << std::endl;
    {
      for(int j=0;j<h;j++)
      {
        std::cout << c[j] << '\t';
      }
      std::cout << '\n';
    }
  }

  void initialize_weights(DataUnit* d)
  {
    if(v==d->h&&h==d->v)
    {
      for(int i=0;i<v;i++)
      {
        for(int j=0;j<h;j++)
        {
          W[i*h+j] = d->W[j*d->h+i];
        }
      }
    }
  }

  void initialize_weights(DataUnit* d1,DataUnit* d2)
  {
    if(v==d1->h+d2->h&&d1->v==h&&d2->v==h)
    {
      std::cout << "initialize bottleneck" << std::endl;
      //char ch;
      //std::cin >> ch;
      int j=0;
      for(int k=0;j<d1->h;j++,k++)
      {
        for(int i=0;i<d1->v;i++)
        {
          W[i*h+j] = d1->W[k*d1->h+i];
        }
      }
      for(int k=0;j<d1->h+d2->h;j++,k++)
      {
        for(int i=0;i<d2->v;i++)
        {
          W[i*h+j] = d2->W[k*d2->h+i];
        }
      }
    }
  }

};

// Multi Layer RBM
//
//  Auto-encoder
//
//          [***]
//         /     \
//     [*****] [*****]
//       /         \
// [********]   [********]
//   inputs      outputs
//
struct mRBM
{
  long in_samp;
  long out_samp;
  bool model_ready;
  std::vector<DataUnit*>  input_branch;
  std::vector<DataUnit*> output_branch;
  DataUnit* bottle_neck;
  void addInputDatUnit(long v,long h)
  {
    DataUnit * unit = new DataUnit(v,h);
    input_branch.push_back(unit);
  }
  void addOutputDatUnit(long v,long h)
  {
    output_branch.push_back(new DataUnit(v,h));
  }
  void addBottleNeckDatUnit(long v,long h)
  {
    bottle_neck = new DataUnit(v,h);
  }
  void construct(std::vector<long> input_num,std::vector<long> output_num,long bottle_neck_num)
  {
    for(long i=0;i+1<input_num.size();i++)
    {
      input_branch.push_back(new DataUnit(input_num[i],input_num[i+1]));
    }
    for(long i=0;i+1<output_num.size();i++)
    {
      output_branch.push_back(new DataUnit(output_num[i],output_num[i+1]));
    }
    bottle_neck = new DataUnit(input_num[input_num.size()-1]+output_num[output_num.size()-1],bottle_neck_num);
  }
  mRBM(long _in_samp,long _out_samp)
  {
    in_samp = _in_samp;
    out_samp = _out_samp;
    model_ready = false;
    bottle_neck = NULL;
  }
  void copy(double * X,double * Y,long num)
  {
    for(long i=0;i<num;i++)
    {
      Y[i] = X[i];
    }
  }
  void model_simple(long sample,double * in,double * out)
  {
    double * X = NULL;
    double * Y = NULL;
    X = new double[in_samp];
    for(long i=0;i<in_samp;i++)
    {
      X[i] = in[sample*in_samp+i];
    }
    for(long i=0;i<input_branch.size();i++)
    {
      Y = new double[input_branch[i]->h];
      input_branch[i]->rbm->vis2hid_simple(X,Y);
      delete [] X;
      X = NULL;
      X = new double[input_branch[i]->h];
      copy(Y,X,input_branch[i]->h);
      delete [] Y;
      Y = NULL;
    }
    double * X_bottleneck = NULL;
    X_bottleneck = new double[bottle_neck->h];
    for(long i=0;i<in_samp;i++)
    {
      X_bottleneck[i] = X[i];
    }
    for(long i=in_samp;i<bottle_neck->h;i++)
    {
      X_bottleneck[i] = 0;
    }
    delete [] X;
    X = NULL;
    {
      double * Y_bottleneck = NULL;
      Y_bottleneck = new double[bottle_neck->h];
      bottle_neck->rbm->vis2hid_simple(X_bottleneck,Y_bottleneck);
      bottle_neck->rbm->hid2vis_simple(Y_bottleneck,X_bottleneck);
      delete [] Y_bottleneck;
      Y_bottleneck = NULL;
      Y = new double[out_samp];
      for(long i=in_samp,k=0;i<bottle_neck->h;i++,k++)
      {
        Y[k] = X_bottleneck[i];
      }
      delete [] X_bottleneck;
      X_bottleneck = NULL;
    }
    for(long j=0;j<out_samp;j++)
    {
      out[sample*out_samp+j] = Y[j];//(Y[j]+1e-5)/(Y_max+1e-5);
    }
    for(long i=output_branch.size()-1;i>=0;i--)
    {
      X = new double[output_branch[i]->v];
      output_branch[i]->rbm->hid2vis_simple(Y,X);
      delete [] Y;
      Y = NULL;
      Y = new double[output_branch[i]->v];
      copy(X,Y,output_branch[i]->v);
      delete [] X;
      X = NULL;
      for(long j=0;j<output_branch[i]->v;j++)
      {
        out[sample*output_branch[i]->v+j] = Y[j];
      }
    }
    delete [] Y;
    Y = NULL;
  }
  double ** model(long sample,double * in)
  {
    double ** out = new double*[20];
    for(int i=0;i<20;i++)out[i]=new double[bottle_neck->v];
    for(int l=0;l<20;l++)
    for(int i=0;i<bottle_neck->v;i++)
    out[l][i]=0;
    //std::cout << "model:\t\t";
    //for(int i=0;i<input_branch[0]->v;i++)
    //std::cout << in[sample*input_branch[0]->v+i] << '\t';
    //std::cout << '\n';
    long layer = 0;
    double * X = NULL;
    double * Y = NULL;
    X = new double[in_samp];
    for(long i=0;i<in_samp;i++)
    {
      X[i] = in[sample*in_samp+i];
    }
    for(long i=0;i<in_samp;i++)
    {
      out[layer][i] = X[i];
    }
    //std::cout << "out:\t\t";
    //for(int i=0;i<input_branch[0]->v;i++)
    //std::cout << out[layer][i] << '\t';
    //std::cout << '\n';
    layer++;
    //std::cout << "input_branch size:" << input_branch.size() << std::endl;
    for(long i=0;i<input_branch.size();i++)
    {
      Y = new double[input_branch[i]->h];
      input_branch[i]->rbm->vis2hid_simple(X,Y);
      delete [] X;
      X = NULL;
      X = new double[input_branch[i]->h];
      copy(Y,X,input_branch[i]->h);
      delete [] Y;
      Y = NULL;
      for(long j=0;j<input_branch[i]->h;j++)
      {
        out[layer][j] = X[j];
      }
      layer++;
    }
    double * X_bottleneck = NULL;
    X_bottleneck = new double[bottle_neck->h];
    for(long i=0;i<in_samp;i++)
    {
      X_bottleneck[i] = X[i];
    }
    for(long i=in_samp;i<bottle_neck->h;i++)
    {
      X_bottleneck[i] = 0;
    }
    delete [] X;
    X = NULL;
    {
      double * Y_bottleneck = NULL;
      Y_bottleneck = new double[bottle_neck->h];
      bottle_neck->rbm->vis2hid_simple(X_bottleneck,Y_bottleneck);
      for(long j=0;j<bottle_neck->v;j++)
      {
        out[layer][j] = X_bottleneck[j];
      }
      layer++;
      for(long j=0;j<bottle_neck->v;j++)
      {
        out[layer][j] = Y_bottleneck[j];
      }
      layer++;
      bottle_neck->rbm->hid2vis_simple(Y_bottleneck,X_bottleneck);
      for(long j=0;j<bottle_neck->v;j++)
      {
        out[layer][j] = X_bottleneck[j];
      }
      layer++;
      delete [] Y_bottleneck;
      Y_bottleneck = NULL;
      Y = new double[out_samp];
      for(long i=in_samp,k=0;i<bottle_neck->h;i++,k++)
      {
        Y[k] = X_bottleneck[i];
      }
      delete [] X_bottleneck;
      X_bottleneck = NULL;
    }
    //double Y_max = 0;
    //for(long j=0;j<bottle_neck->v-input_branch[input_branch.size()-1]->v;j++)
    //{
    //  if(Y[j]>Y_max)Y_max = Y[j];
    //}
    for(long j=0;j<out_samp;j++)
    {
      out[layer][j] = Y[j];//(Y[j]+1e-5)/(Y_max+1e-5);
    }
    layer++;
    for(long i=output_branch.size()-1;i>=0;i--)
    {
      X = new double[output_branch[i]->v];
      output_branch[i]->rbm->hid2vis_simple(Y,X);
      delete [] Y;
      Y = NULL;
      Y = new double[output_branch[i]->v];
      copy(X,Y,output_branch[i]->v);
      delete [] X;
      X = NULL;
      for(long j=0;j<output_branch[i]->v;j++)
      {
        out[layer][j] = Y[j];
      }
      layer++;
    }
    //for(long i=0;i<output_branch[0]->v;i++)
    //{
    //  out[layer][i] = Y[i];
    //}
    delete [] Y;
    Y = NULL;
    return out;
  }
  void train(long in_num,long out_num,long n_samp,long total_n,long n_cd,double epsilon,double * in,double * out)
  {
    double * X = NULL;
    double * Y = NULL;
    double * IN = NULL;
    double * OUT = NULL;
    X = new double[in_num*n_samp];
    IN = new double[in_num*n_samp];
    for(long i=0;i<in_num*n_samp;i++)
    {
      X[i] = in[i];
      IN[i] = in[i];
    }
    for(long i=0;i<input_branch.size();i++)
    {
      if(i>0)input_branch[i]->initialize_weights(input_branch[i-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      input_branch[i]->train(X,n_samp,total_n,n_cd,epsilon,input_branch[i]->h);
      Y = new double[input_branch[i]->h*n_samp];
      input_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      //std::cout << "X init:" << in_num*n_samp << "    " << "X fin:" << input_branch[i]->h*n_samp << std::endl;
      X = new double[input_branch[i]->h*n_samp];
      copy(Y,X,input_branch[i]->h*n_samp);
      copy(Y,IN,input_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    X = new double[out_num*n_samp];
    OUT = new double[in_num*n_samp];
    for(long i=0;i<out_num*n_samp;i++)
    {
      X[i] = out[i];
      OUT[i] = out[i];
    }
    for(long i=0;i<output_branch.size();i++)
    {
      if(i>0)output_branch[i]->initialize_weights(output_branch[i-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      output_branch[i]->train(X,n_samp,total_n,n_cd,epsilon,input_branch[i]->h);
      Y = new double[output_branch[i]->h*n_samp];
      output_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      X = new double[output_branch[i]->h*n_samp];
      copy(Y,X,output_branch[i]->h*n_samp);
      copy(Y,OUT,output_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    if(bottle_neck!=NULL)
    {
      X = new double[bottle_neck->h*n_samp];
      for(long s=0;s<n_samp;s++)
      {
        long i=0;
        for(long k=0;i<in_num&&k<in_num;i++,k++)
        {
          X[s*(in_num+out_num)+i] = IN[s*in_num+k];
        }
        for(long k=0;i<in_num+out_num&&k<out_num;i++,k++)
        {
          X[s*(in_num+out_num)+i] = OUT[s*out_num+k];
        }
      }
      //bottle_neck->initialize_weights(input_branch[input_branch.size()-1],output_branch[output_branch.size()-1]); // initialize weights to transpose of previous layer weights M_i -> W = M_{i-1} -> W ^ T
      bottle_neck->train(X,n_samp,total_n,n_cd,epsilon,in_num+out_num);
      delete [] X;
      X = NULL;
    }
    delete [] IN;
    IN = NULL;
    delete [] OUT;
    OUT = NULL;
    model_ready = true;
  }
  double compare(long sample,double * a,double * b)
  {
    double sum_a = 0;
    double sum_b = 0;
    for(int i=0;i<out_samp;i++)
    {
      sum_a += a[sample*out_samp+i];
      sum_b += b[sample*out_samp+i];
    }
    for(int i=0;i<out_samp;i++)
    {
      a[sample*out_samp+i] = (a[sample*out_samp+i]+1e-10)/(sum_a+1e-5);
      b[sample*out_samp+i] = (b[sample*out_samp+i]+1e-10)/(sum_b+1e-5);
    }
    double score = 0;
    for(int i=0;i<out_samp;i++)
    {
      //score += ((a[sample*out_samp+i]>0.5&&b[sample*out_samp+i]>0.5)||(a[sample*out_samp+i]<0.5&&b[sample*out_samp+i]<0.5))?1:0;
      score += (a[sample*out_samp+i]*b[sample*out_samp+i]);
      score += (1-a[sample*out_samp+i])*(1-b[sample*out_samp+i]);
    }
    return score/out_samp;
  }
  void compare_all(long num,double * in,double * out)
  {
    double score = 0;
    for(long i=0;i<num;i++)
    {
      score += (compare(i,in,out)-score)/(1+i);
      for(int j=0;j<out_samp;j++)
      {
        std::cout << ((in[i*out_samp+j]>0.5)?"1":"0") << ":" << ((out[i*out_samp+j]>0.5)?"1":"0") << "\t";
      }
    }
    std::cout << std::endl;
    std::cout << "score:" << score << std::endl;
    //char ch;
    //std::cin >> ch;
  }
  void model_all(long num,double * in,double * out)
  {
    for(long i=0;i<num;i++)
    {
      model_simple(i,in,out);
    }
  }
};

struct mrbm_params
{

  long batch_iter;
  long num_batch;
  long total_n;
  long n;
  double epsilon;
  long n_iter;
  long n_cd;

  long v;
  long h;

  std::vector<long> input_sizes;

  std::vector<long> output_sizes;

  std::vector<long> input_iters;

  std::vector<long> output_iters;

  long bottleneck_iters;

  mrbm_params(int _v,int _h,long n_iter,long n_batch,long n_samples,double c_epsilon)
  {

    v = _v;
    h = _h;

    n_cd = 1;
    num_batch = n_batch;
    batch_iter = 1;
    n = n_samples;
    total_n = n_samples;
    epsilon = c_epsilon;

    input_sizes.push_back(v);

    output_sizes.push_back(h);

    for(int i=0;i+1<input_sizes.size();i++)
        input_iters.push_back(n_iter);

    for(int i=0;i+1<output_sizes.size();i++)
        output_iters.push_back(n_iter);

    bottleneck_iters = n_iter;

  }
};

mRBM * mrbm = NULL;

void run_mrbm(mrbm_params p,double * dat_in,double * dat_out)
{
  int cd = 0;
  for(int iter = 0; iter < 10; iter++)
  {
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    if(mrbm == NULL)
    {
      mrbm = new mRBM(p.input_sizes[0],p.output_sizes[0]);
      for(long i=0;i+1<p.input_sizes.size();i++)
      {
        mrbm->input_branch.push_back(new DataUnit(p.input_sizes[i],p.input_sizes[i+1],p.input_iters[i]));
      }
      for(long i=0;i+1<p.output_sizes.size();i++)
      {
        mrbm->output_branch.push_back(new DataUnit(p.output_sizes[i],p.output_sizes[i+1],p.output_iters[i]));
      }
      long bottle_neck_num = (p.input_sizes[p.input_sizes.size()-1]+p.output_sizes[p.output_sizes.size()-1]);
      mrbm->bottle_neck = new DataUnit(p.input_sizes[p.input_sizes.size()-1]+p.output_sizes[p.output_sizes.size()-1],bottle_neck_num,p.bottleneck_iters);
    }
    mrbm->train(p.v,p.h,p.num_batch,p.total_n,(int)(p.n_cd+cd),p.epsilon/(1+0.0*cd),dat_in,dat_out);
    cd ++;
  }
}

void test_xor()
{
  std::vector<long> nodes;
  nodes.push_back(2); // inputs
  nodes.push_back(3); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  Perceptron<double> perceptron(nodes);
  double in_dat[8];
  in_dat[0] = 0;
  in_dat[1] = 0;
  in_dat[2] = 0;
  in_dat[3] = 1;
  in_dat[4] = 1;
  in_dat[5] = 0;
  in_dat[6] = 1;
  in_dat[7] = 1;
  double out_dat[4];
  out_dat[0] = 0;
  out_dat[1] = 1;
  out_dat[2] = 1;
  out_dat[3] = 0;
  perceptron.train(0,100,100000,4,2,in_dat,1,out_dat);
}

void test_spiral()
{
  std::vector<long> nodes;
  nodes.push_back(2); // inputs
  nodes.push_back(10); // hidden layer
  nodes.push_back(10); // hidden layer
  nodes.push_back(10); // hidden layer
  nodes.push_back(10); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  perceptron = new Perceptron<double>(nodes);
  perceptron->epsilon = .1;
  perceptron->sigmoid_type = 0;
  int num_pts = 100;
  int num_iters = 1000;
  while(true)
  {
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

    int num_pts = 100;
     in_dat = new double[num_pts*2];
    out_dat = new double[num_pts];
    double R;
    for(int pt=0;pt<num_pts;pt+=2)
    {
      R = (double)pt/num_pts;
       in_dat[pt*2  ] =  R*cos(4*M_PI*R);
       in_dat[pt*2+1] =  R*sin(4*M_PI*R);
       in_dat[pt*2+2] = -R*cos(4*M_PI*R);
       in_dat[pt*2+3] = -R*sin(4*M_PI*R);
      out_dat[pt  ] = 1;
      out_dat[pt+1] = 0;
    }

    perceptron->train(perceptron->sigmoid_type,perceptron->epsilon,num_iters,num_pts,2,in_dat,1,out_dat);
    delete []  in_dat;
    delete [] out_dat;
    //num_pts = (int)(2*num_pts);
    //num_iters = (int)(2*num_iters);
  }
}

void test_func()
{
  std::vector<long> nodes;
  nodes.push_back(2); // inputs
  nodes.push_back(150); // hidden layer
  nodes.push_back(150); // hidden layer
  nodes.push_back(150); // hidden layer
  nodes.push_back(150); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  perceptron = new Perceptron<double>(nodes);
  perceptron->epsilon = .1;
  perceptron->sigmoid_type = 0;
  int num_pts = 100;
  int num_iters = 1000;
  while(true)
  {
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
     in_dat = new double[num_pts*2];
    out_dat = new double[num_pts];
    double R;
    for(int pt=0;pt<num_pts;pt++)
    {
      in_dat[2*pt  ] = -1+2*((rand()%10000)/10000.0f);
      in_dat[2*pt+1] = -1+2*((rand()%10000)/10000.0f);
      //out_dat[pt] = (sqrt(in_dat[2*pt]*in_dat[2*pt] + in_dat[2*pt+1]*in_dat[2*pt+1]) < 0.9)?1:0;
      std::complex<double> x(1.5*in_dat[2*pt],1.5*in_dat[2*pt+1]);
      std::complex<double> c(0,0);
      out_dat[pt] = 0;
      for(int iter = 0; iter < 1000; iter++)
      {
        //std::cout << iter << "\t" << real(c) << "\t" << imag(c) << std::endl;
        c = x + c*c;
        if(norm(c)>3)
        {
          out_dat[pt] = 1;
          break;
        }
      }
      //std::cout << out_dat[pt] << std::endl;
    }
    perceptron->train(perceptron->sigmoid_type,perceptron->epsilon,num_iters,num_pts,2,in_dat,1,out_dat);
    delete []  in_dat;
    delete [] out_dat;
    num_pts = (int)(2*num_pts);
    num_iters = (int)(2*num_iters);
  }
}

#if 0
void test_boltzman()
{
  int num_inputs = 2;
  int num_outputs = 1;
  int num_elems = 4;
  double * in  = new double[num_inputs *num_elems];
  double * out = new double[num_outputs*num_elems];
  in[0] = 0;
  in[1] = 0;
  in[2] = 0;
  in[3] = 1;
  in[4] = 1;
  in[5] = 0;
  in[6] = 1;
  in[7] = 1;
  out[0] = 0;
  out[1] = 1;
  out[2] = 1;
  out[3] = 0;

  {
    mrbm_params params(num_inputs,num_outputs,1000,num_elems-1,num_elems,1);
    boost::thread * thr ( new boost::thread ( run_mrbm
                                            , params
                                            , in
                                            , out
                                            ) 
                        );
    thr -> join();
  }

  
  long num_iters = 1000000;
  std::vector<long> nodes;
  nodes.push_back(num_inputs); // inputs
  nodes.push_back(num_inputs+num_outputs); // hidden layer
  nodes.push_back(num_outputs); // output layer
  nodes.push_back(num_outputs); // outputs
  perceptron = new Perceptron<double>(nodes);
  perceptron->epsilon = 0.01;
  perceptron->sigmoid_type = 0;
  
  int layer = 0;
  //for(int k=0;k<mrbm->input_branch.size();k++)
  //{
  //  {
  //    for(int i=0;i<mrbm->input_branch[k]->v;i++)
  //    {
  //      for(int j=0;j<mrbm->input_branch[k]->h;j++)
  //      {
  //        perceptron->weights_neuron[layer][i*h+j] = mrbm->input_branch[k]->W[j*d->h+i];
  //      }
  //      perceptron->weights_bias[layer][i] = mrbm->input_branch[k]->b[i];
  //    }
  //  }
  //  layer++;
  //}

  /*
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  W:
  0.24421 0.313074  
  0.0191251 -0.461918 
  b:
  -0.274931 
  0.163652  
  c:
  0.433964  -0.412187 
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  W:
  -0.766262 
  -0.606967 

  b:
  0.678276  -0.323935 
  W:
  -0.355083 0.736694  

  b:
  -0.052887 
  */

  //mrbm->bottle_neck->W[0] = 0.24421;
  //mrbm->bottle_neck->W[1] = 0.313074;
  //mrbm->bottle_neck->W[2] = 0.0191251;
  //mrbm->bottle_neck->W[3] =-0.461918;

  //mrbm->bottle_neck->b[0] =-0.274931;
  //mrbm->bottle_neck->b[1] = 0.163652;

  //mrbm->bottle_neck->c[0] = 0.433964;
  //mrbm->bottle_neck->c[1] =-0.412187;
  //
  //std::cout << '[' << nodes[layer+1] << '\t' << nodes[layer] << ']' << std::endl;
  //std::cout << '[' << mrbm->bottle_neck->v << '\t' << mrbm->bottle_neck->h << ']' << std::endl;
  //void vis2hid_worker(const double * X,double * H,long h,long v,double * c,double * W,std::vector<long> const & vrtx)
  //{
  //  for(long t=0;t<vrtx.size();t++)
  //  {
  //    long k = vrtx[t];
  //    for(long j=0;j<h;j++)
  //    {
  //      H[k*h+j] = c[j]; 
  //      for(long i=0;i<v;i++)
  //      {
  //        H[k*h+j] += W[i*h+j] * X[k*v+i];
  //      }
  //      H[k*h+j] = 1.0f/(1.0f + exp(-H[k*h+j]));
  //    }
  //  }
  //}
  
  {
    {
      for(int i=0;i<nodes[layer+1];i++)
      {
        for(int j=0;j<nodes[layer];j++)
        {
          perceptron->weights_neuron[layer][i][j] = mrbm->bottle_neck->W[i*mrbm->bottle_neck->h+j];
        }
        perceptron->weights_bias[layer][i] = (i<num_inputs)?mrbm->bottle_neck->b[i]:0;
      }
    }
    layer++;
  }
  //std::cout << '[' << nodes[layer+1] << '\t' << nodes[layer] << ']' << std::endl;
  //std::cout << '[' << mrbm->bottle_neck->v << '\t' << mrbm->bottle_neck->h << ']' << std::endl;
  {
    {
      for(int i=0;i<nodes[layer+1];i++)
      {
        for(int j=0;j<nodes[layer];j++)
        {
          perceptron->weights_neuron[layer][i][j] = mrbm->bottle_neck->W[j*mrbm->bottle_neck->h+i+num_inputs];
        }
        perceptron->weights_bias[layer][i] = mrbm->bottle_neck->c[i+num_inputs];
      }
    }
    layer++;
  }
  
  //for(int k=0;k<mrbm->output_branch.size();k++)
  //{
  //  {
  //    for(int i=0;i<mrbm->output_branch[k]->v;i++)
  //    {
  //      for(int j=0;j<mrbm->output_branch[k]->h;j++)
  //      {
  //        perceptron->weights_neuron[layer][i*h+j] = mrbm->output_branch[k]->W[j*d->h+i];
  //      }
  //      perceptron->weights_bias[layer][i] = mrbm->output_branch[k]->c[i];
  //    }
  //  }
  //  layer++;
  //}

  if(mrbm != NULL)
  {
    {
      for(long i=0;i+1<mrbm->input_branch.size();i++)
      {
        mrbm->input_branch[i]->print();
      }
      mrbm->bottle_neck->print();
      for(long i=0;i+1<mrbm->output_branch.size();i++)
      {
        mrbm->output_branch[i]->print();
      }
    }
  }
  
  if(perceptron != NULL)
  {
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
    for(long layer = 0;layer < nodes.size()-2;layer++)
    {
      std::cout << "W:" << std::endl;
      for(long i=0;i<nodes[layer+1];i++)
      {
        for(long j=0;j<nodes[layer];j++)
        {
          std::cout << perceptron->weights_neuron[layer][i][j] << '\t';
        }
        std::cout << '\n';
      }
      std::cout << '\n';
      std::cout << "b:" << std::endl;
      for(long i=0;i<nodes[layer+1];i++)
      {
        std::cout << perceptron->weights_bias[layer][i] << '\t';
      }
      std::cout << '\n';
    }
  }
  char ch;
  std::cin >> ch;

  perceptron->train(perceptron->sigmoid_type,perceptron->epsilon,num_iters,num_elems,num_inputs,in,num_outputs,out);

  if(mrbm != NULL)
  {
    {
      for(long i=0;i+1<mrbm->input_branch.size();i++)
      {
        mrbm->input_branch[i]->print();
      }
      mrbm->bottle_neck->print();
      for(long i=0;i+1<mrbm->output_branch.size();i++)
      {
        mrbm->output_branch[i]->print();
      }
    }
  }

  if(perceptron != NULL)
  {
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
    for(long layer = 0;layer < nodes.size()-2;layer++)
    {
      std::cout << "W:" << std::endl;
      for(long i=0;i<nodes[layer+1];i++)
      {
        for(long j=0;j<nodes[layer];j++)
        {
          std::cout << perceptron->weights_neuron[layer][i][j] << '\t';
        }
        std::cout << '\n';
      }
      std::cout << '\n';
      std::cout << "b:" << std::endl;
      for(long i=0;i<nodes[layer+1];i++)
      {
        std::cout << perceptron->weights_bias[layer][i] << '\t';
      }
      std::cout << '\n';
    }
  }
}
#endif

void test_boltzman()
{
  int num_inputs = 2;
  int num_outputs = 1;
  int num_hidden = 3;
  int num_elems = 4;
  double * in  = new double[num_inputs *num_elems];
  double * out = new double[num_outputs*num_elems];
  in[0] = 0;
  in[1] = 0;
  in[2] = 0;
  in[3] = 1;
  in[4] = 1;
  in[5] = 0;
  in[6] = 1;
  in[7] = 1;
  out[0] = 0;
  out[1] = 1;
  out[2] = 1;
  out[3] = 0;

  RBM * rbm = new RBM(num_inputs,num_hidden,num_elems,in);
  for(int i=0;i<10000;i++)
  {
    rbm->init(0);
    rbm->cd(1,0.1,0);
  }
  double * hid = new double[num_hidden*num_elems];
  rbm->vis2hid(in,hid);
  RBM * rbm2 = new RBM(num_hidden,num_outputs,num_elems,hid);
  for(int i=0;i<10000;i++)
  {
    rbm2->init(0);
    rbm2->cd(1,0.1,0);
  }

  long num_iters = 100000;
  std::vector<long> nodes;
  nodes.push_back(num_inputs); // inputs
  nodes.push_back(num_hidden); // hidden layer
  nodes.push_back(num_outputs); // output layer
  nodes.push_back(num_outputs); // outputs
  perceptron = new Perceptron<double>(nodes);
  perceptron->epsilon = 0.1;
  perceptron->sigmoid_type = 0;
  
  int layer = 0;
  {
    {
      for(int i=0;i<nodes[layer+1];i++)
      {
        for(int j=0;j<nodes[layer];j++)
        {
          perceptron->weights_neuron[layer][i][j] = rbm->W[j*rbm->h+i];
        }
        perceptron->weights_bias[layer][i] = rbm->c[i];
      }
    }
    layer++;
  }
  //{
  //  {
  //    for(int i=0;i<nodes[layer+1];i++)
  //    {
  //      for(int j=0;j<nodes[layer];j++)
  //      {
  //        perceptron->weights_neuron[layer][i][j] = rbm2->W[j*rbm2->h+i];
  //      }
  //      perceptron->weights_bias[layer][i] = rbm2->c[i];
  //    }
  //  }
  //  layer++;
  //}

  if(rbm != NULL)
  {
    {
      {
        rbm->print();
      }
    }
  }
  if(rbm2 != NULL)
  {
    {
      {
        rbm2->print();
      }
    }
  }
  
  if(perceptron != NULL)
  {
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
    for(long layer = 0;layer < nodes.size()-2;layer++)
    {
      std::cout << "W:" << std::endl;
      for(long i=0;i<nodes[layer+1];i++)
      {
        for(long j=0;j<nodes[layer];j++)
        {
          std::cout << perceptron->weights_neuron[layer][i][j] << '\t';
        }
        std::cout << '\n';
      }
      std::cout << '\n';
      std::cout << "b:" << std::endl;
      for(long i=0;i<nodes[layer+1];i++)
      {
        std::cout << perceptron->weights_bias[layer][i] << '\t';
      }
      std::cout << '\n';
    }
  }
  char ch;
  std::cin >> ch;

  perceptron->train(perceptron->sigmoid_type,perceptron->epsilon,num_iters,num_elems,num_inputs,in,num_outputs,out);

  if(rbm != NULL)
  {
    {
      {
        rbm->print();
      }
    }
  }
  if(rbm2 != NULL)
  {
    {
      {
        rbm2->print();
      }
    }
  }
  
  if(perceptron != NULL)
  {
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
    for(long layer = 0;layer < nodes.size()-2;layer++)
    {
      std::cout << "W:" << std::endl;
      for(long i=0;i<nodes[layer+1];i++)
      {
        for(long j=0;j<nodes[layer];j++)
        {
          std::cout << perceptron->weights_neuron[layer][i][j] << '\t';
        }
        std::cout << '\n';
      }
      std::cout << '\n';
      std::cout << "b:" << std::endl;
      for(long i=0;i<nodes[layer+1];i++)
      {
        std::cout << perceptron->weights_bias[layer][i] << '\t';
      }
      std::cout << '\n';
    }
  }

}

int min(int a,int b)
{
    return (a>b)?b:a;
}

void test_boltzman_multilayer()
{
  int num_inputs  = (nx+ny);
  int num_outputs = 1;
  std::vector<int> num_hidden;
  num_hidden.push_back((nx*ny));
  num_hidden.push_back((nx*ny));
  num_hidden.push_back((nx*ny));
  num_hidden.push_back((nx*ny));
  int num_elems = 10000;
  int num_approx = 3;
  int num_iters = 100;
  double eps = 1.0;
  double * in  = new double[num_inputs *num_elems];
  double * out = new double[num_outputs*num_elems];
  for(long x=0;x<num_elems;x++)
  {
    double R;
    double X = ((rand()%10000)/10000.0f);
    double Y = ((rand()%10000)/10000.0f);
    double dX = -1+2*X;
    double dY = -1+2*Y;
    int bX = pow(2,nx)*X;
    int bY = pow(2,ny)*Y;
    for(long i=0;i<nx;i++)
    {
      in[x*num_inputs+i] = bX%2==0;
      bX/=2;
    }
    for(long i=nx;i<nx+ny;i++)
    {
      in[x*num_inputs+i] = bY%2==0;
      bY/=2;
    }
    std::complex<double> y(1.5*dX,1.5*dY);
    std::complex<double> c(0,0);
    out[x] = 0;
    for(int iter = 0; iter < 1000; iter++)
    {
      c = y + c*c;
      if(norm(c)>3)
      {
        out[x] = 1;
        break;
      }
    }
  }

  //std::vector < RBM * > rbm;
  //{
  //  std::cout << num_inputs << "->" << num_hidden[0] << std::endl;
  //  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  //  rbm.push_back(new RBM(num_inputs,num_hidden[0],num_elems,in));
  //  for(int j=1;j<=num_approx;j++)
  //  {
  //    std::cout << j << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  //    for(int i=0;i<num_iters;i++)
  //    {
  //      rbm[rbm.size()-1]->init(0);
  //      rbm[rbm.size()-1]->cd(j,eps/j,0);
  //    }
  //  }
  //  double ** hid = new double*[num_hidden.size()];
  //  for(int i=0;i<num_hidden.size();i++)
  //  {
  //      hid[i] = new double[num_hidden[i]*num_elems];
  //  }
  //  rbm[rbm.size()-1]->vis2hid(in,hid[0]);
  //  for(int i=0;i<num_hidden[0]*num_elems;i++)
  //      hid[0][i] = (hid[0][i]>0.5)?1:0;
  //  for(int i=0;i<3*num_hidden[0];i++)
  //      std::cout << hid[0][i] << '\t';
  //  std::cout << std::endl;
  //  for(int l=0;l+1<num_hidden.size();l++)
  //  {
  //    std::cout << num_hidden[l] << "->" << num_hidden[l+1] << std::endl;
  //    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  //    rbm.push_back(new RBM(num_inputs,num_hidden[l],num_hidden[l+1],hid[l]));
  //    for(int j=1;j<=num_approx;j++)
  //    {
  //      std::cout << j << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  //      for(int i=0;i<num_iters;i++)
  //      {
  //        rbm[rbm.size()-1]->init(0);
  //        rbm[rbm.size()-1]->cd(j,eps/j,0);
  //      }
  //    }
  //    rbm[rbm.size()-1]->vis2hid(hid[l],hid[l+1]);
  //    for(int i=0;i<num_hidden[l+1]*num_elems;i++)
  //        hid[l+1][i] = (hid[l+1][i]>0.5)?1:0;
  //    for(int i=0;i<3*num_hidden[l+1];i++)
  //        std::cout << hid[l+1][i] << '\t';
  //    std::cout << std::endl;
  //  }
  //  std::cout << num_hidden[num_hidden.size()-1] << "->" << num_outputs << std::endl;
  //  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  //  rbm.push_back(new RBM(num_hidden[num_hidden.size()-1],num_outputs,num_elems,hid[num_hidden.size()-1]));
  //  for(int j=1;j<=num_approx;j++)
  //  {
  //    std::cout << j << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
  //    for(int i=0;i<num_iters;i++)
  //    {
  //      rbm[rbm.size()-1]->init(0);
  //      rbm[rbm.size()-1]->cd(j,eps/j,0);
  //    }
  //  }
  //}

  long num_ann_iters = 100000;
  std::vector<long> nodes;
  nodes.push_back(num_inputs); // inputs
  for(int h=0;h<num_hidden.size();h++)
    nodes.push_back(num_hidden[h]); // hidden layer
  nodes.push_back(num_outputs); // output layer
  nodes.push_back(num_outputs); // outputs
  perceptron = new Perceptron<double>(nodes);
  perceptron->epsilon = 0.1;
  perceptron->alpha = 0.1;
  perceptron->sigmoid_type = 0;
  
  //int layer = 0;
  //for(int l=0;l+1<rbm.size();l++)
  //{
  //  {
  //    for(int i=0;i<nodes[layer+1];i++)
  //    {
  //      for(int j=0;j<nodes[layer];j++)
  //      {
  //        perceptron->weights_neuron[layer][i][j] = rbm[l]->W[j*rbm[l]->h+i];
  //      }
  //      perceptron->weights_bias[layer][i] = rbm[l]->c[i];
  //    }
  //  }
  //  layer++;
  //}

  perceptron->train(perceptron->sigmoid_type,perceptron->epsilon,num_ann_iters,num_elems,num_inputs,in,num_outputs,out);

}

int main(int argc,char ** argv)
{
  srand(time(NULL));
  //test_xor();
  //boost::thread * th = new boost::thread(test_spiral);
  //boost::thread * th = new boost::thread(test_func);
  //boost::thread * th = new boost::thread(test_boltzman);
  boost::thread * th = new boost::thread(test_boltzman_multilayer);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("multi-layer nn tests");
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutIdleFunc(idle);
  init();
  glutMainLoop();
  return 0;
}

