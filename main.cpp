#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <boost/thread.hpp>

template<typename T>
struct Perceptron
{

    T perror;

    T *** weights_neuron;
    T **  weights_bias;
    T **  activation_values;
    T **  activation_values1;
    T **  deltas;
    long * number_of_nodes;

    long n_inputs;
    long n_outputs;
    long n_layers;
    std::vector<long> n_nodes;

    // std::vector<long> nodes;
    // nodes.push_back(2); // inputs
    // nodes.push_back(3); // hidden layer
    // nodes.push_back(1); // output layer
    // nodes.push_back(1); // outputs
    Perceptron(std::vector<long> p_nodes)
    {

        perror = 1e10;

        n_nodes = p_nodes;
        n_inputs = n_nodes[0];
        n_outputs = n_nodes[n_nodes.size()-1];
        n_layers = n_nodes.size()-2; // first and last numbers and output and input dimensions, so we have n-2 layers

        weights_neuron = new T**[n_layers];
        weights_bias = new T*[n_layers];
        number_of_nodes  = new long[n_layers];
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
            number_of_nodes [layer] = n_nodes[layer+1];
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

    }

    T sigmoid1(T x)
    {
        return 1.0f / (1.0f + exp(-x));
    }

    T dsigmoid1(T x)
    {
        return (1.0f - x)*x;
    }

    T sigmoid2(T x)
    {
        return log(1+exp(0.05*x));
    }

    T dsigmoid2(T x)
    {
        return 0.05/(1+exp(-0.05*x));
    }

    T sigmoid(T x)
    {
        return (perror<4000)?sigmoid2(x):sigmoid1(x);
    }

    T dsigmoid(T x)
    {
        return (perror<4000)?dsigmoid2(x):dsigmoid1(x);
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
                activation_values1[layer+1][i] = sigmoid(sum);
            }
        }
        long last_layer = n_nodes.size()-2;
        for(long i=0;i<n_labels;i++)
        {
            labels[i] = activation_values1[last_layer][i];
        }
        return labels;
    }

    void train(T epsilon,long n_iterations,long n_elements,long n_variables,T * variables,long n_labels, T * labels)
    {
        if(n_variables != n_nodes[0]){std::cout << "error 789437248932748293" << std::endl;exit(0);}
        for(long iter = 0; iter < n_iterations; iter++)
        {
            perror = 1e10;
            T error = 0;
            for(long n=0,k=0,t=0;n<n_elements;n++)
            {
                // initialize input activations
                for(long i=0;i<n_nodes[0];i++,k++)
                {
                    activation_values[0][i] = variables[k];
                }
                // forward propagation
                for(long layer = 0; layer < n_layers; layer++)
                {
                    for(long i=0;i<n_nodes[layer+1];i++)
                    {
                        T sum = weights_bias[layer][i];
                        for(long j=0;j<n_nodes[layer];j++)
                        {
                            sum += activation_values[layer][j] * weights_neuron[layer][i][j];
                        }
                        activation_values[layer+1][i] = sigmoid(sum);
                    }
                }
                long last_layer = n_nodes.size()-2;
                // initialize observed labels
                for(long i=0;i<n_nodes[last_layer];i++,t++)
                {
                    deltas[last_layer][i] = labels[t] - activation_values[last_layer][i];
                    error += fabs(deltas[last_layer][i]);
                }
                // back propagation
                for(long layer = n_layers-1; layer >= 0; layer--)
                {
                    // back propagate deltas
                    for(long i=0;i<n_nodes[layer+1];i++)
                    {
                        deltas[layer][i] = 0;
                        for(long j=0;j<n_nodes[layer+2];j++)
                        {
                            if(layer+1==last_layer)
                            {
                                deltas[layer][i] += dsigmoid(activation_values[layer+1][i])*deltas[layer+1][j];
                            }
                            else
                            {
                                deltas[layer][i] += dsigmoid(activation_values[layer+1][i])*deltas[layer+1][j]*weights_neuron[layer+1][j][i];
                            }
                        }
                    }
                    // biases
                    for(long i=0;i<n_nodes[layer+1];i++)
                    {
                        weights_bias[layer][i] += epsilon * deltas[layer][i];
                    }
                    // neuron weights
                    for(long i=0;i<n_nodes[layer+1];i++)
                    {
                        for(long j=0;j<n_nodes[layer];j++)
                        {
                            weights_neuron[layer][i][j] += epsilon * activation_values[layer][j] * deltas[layer][i];
                        }
                    }
                }
            }
            std::cout << "error=" << error << std::endl;
            perror = error;
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
    default:break;
  }
}

void test_xor()
{
  std::vector<long> nodes;
  nodes.push_back(2); // inputs
  nodes.push_back(3); // hidden layer
  nodes.push_back(2); // hidden layer
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
  perceptron.train(0.1,100000,4,2,in_dat,1,out_dat);
}

void test_spiral()
{
  std::vector<long> nodes;
  nodes.push_back(2); // inputs
  nodes.push_back(5); // hidden layer
  nodes.push_back(5); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  perceptron = new Perceptron<double>(nodes);
  int num_pts = 1000;
   in_dat = new double[num_pts*2];
  out_dat = new double[num_pts];
  double R;
  for(int pt=0;pt<num_pts;pt+=2)
  {
    R = (double)pt/num_pts;
     in_dat[pt*2  ] =  R*cos(pt*0.01);
     in_dat[pt*2+1] =  R*sin(pt*0.01);
     in_dat[pt*2+2] = -R*cos(pt*0.01);
     in_dat[pt*2+3] = -R*sin(pt*0.01);
    out_dat[pt  ] = 1;
    out_dat[pt+1] = 0;
  }
  perceptron->train(0.01,1000000,num_pts,2,in_dat,1,out_dat);
}

void test_func()
{
  std::vector<long> nodes;
  nodes.push_back(2); // inputs
  nodes.push_back(25); // hidden layer
  nodes.push_back(25); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  perceptron = new Perceptron<double>(nodes);
  while(true)
  {
    int num_pts = 100000;
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
    perceptron->train(0.01,1000,num_pts,2,in_dat,1,out_dat);
    delete []  in_dat;
    delete [] out_dat;
  }
}

int main(int argc,char ** argv)
{
  //test_xor();
  //test_spiral();
  boost::thread * th = new boost::thread(test_func);
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

