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

template<typename T>
struct training_info
{
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

    void init()
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

    void globalUpdate()
    {
        for(long layer = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++)
                {
                    weights_neuron[layer][i][j] += partial_weights_neuron[layer][i][j] / n_elements;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                weights_bias[layer][i] += partial_weights_bias[layer][i] / n_elements;
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
                g->partial_weights_bias[layer][i] += g->epsilon * g->deltas[layer+1][i];
                //std::cout << g->partial_weights_bias[layer][i] << '\t';
            }
            //std::cout << std::endl;
            //std::cout << "neuron weights" << std::endl;
            // neuron weights
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                for(long j=0;j<g->n_nodes[layer];j++)
                {
                    g->partial_weights_neuron[layer][i][j] += g->epsilon * g->activation_values[layer][j] * g->deltas[layer+1][i];
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
    int sigmoid_type;

    // std::vector<long> nodes;
    // nodes.push_back(2); // inputs
    // nodes.push_back(3); // hidden layer
    // nodes.push_back(1); // output layer
    // nodes.push_back(1); // outputs
    Perceptron(std::vector<long> p_nodes)
    {

        sigmoid_type = 0;

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
        for(long iter = 0; iter < n_iterations; iter++)
        {
            ierror = 1e10;
            bool init = true;
            perror = 1e10;
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
            for(long thread=0;thread<vrtx.size();thread++)
            {

              g[thread]->n_nodes = n_nodes;
              g[thread]->n_elements = n_elements;
              g[thread]->n_variables = n_variables;
              g[thread]->n_labels = n_labels;
              g[thread]->n_layers = n_layers;
              g[thread]->weights_neuron = weights_neuron;
              g[thread]->weights_bias = weights_bias;
              g[thread]->epsilon = epsilon;
              g[thread]->type = get_sigmoid();

              g[thread]->init();
              threads.push_back(new boost::thread(training_worker<T>,g[thread],vrtx[thread],variables,labels));
            }
            for(long thread=0;thread<vrtx.size();thread++)
            {
              threads[thread]->join();
              delete threads[thread];
            }
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


            std::cout << "type=" << sigmoid_type << "\tepsilon=" << epsilon << '\t' << "error=" << error << std::endl;
            perror = error;
            if(init)
            {
                ierror = error;
                init = false;
            }

        }
    }

};

double *  in_dat = NULL;
double * out_dat = NULL;

Perceptron<double> * perceptron = NULL;

void
drawStuff(void)
{
  double in[2];
  float dx = 2/200.0f;
  float dy = 2/200.0f;
  glBegin(GL_QUADS);
  for (float x = -1; x <= 1; x+=dx) 
  for (float y = -1; y <= 1; y+=dy) 
  {
    in[0] = x;
    in[1] = y;
    double * out = perceptron->model(2,1,in);
    glColor3f(out[0],out[0],out[0]);
    delete out;
    glVertex3f(x   ,y   ,0);
    glVertex3f(x+dx,y   ,0);
    glVertex3f(x+dx,y+dy,0);
    glVertex3f(x   ,y+dy,0);
  }
  glEnd();
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
    case 'a':perceptron->sigmoid_type = (perceptron->sigmoid_type+1)%2; break;
    case 'd':perceptron->sigmoid_type = (perceptron->sigmoid_type+1)%2; break;
    default:break;
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

int main(int argc,char ** argv)
{
  //test_xor();
  boost::thread * th = new boost::thread(test_spiral);
  //boost::thread * th = new boost::thread(test_func);
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

