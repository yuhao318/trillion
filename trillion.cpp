//    n: number of unknown variables. It must be a power of 2
//    m: number of rows in matrix A. It must be a power of 2 and >= n
//    k: number of nonzeros in the optimal solution. It must be a power of 
//       2 and <= n
//    tau: the parameter that scales the l1-norm in the objective function
//    S: the singular values of matrix A. It must be a column vector of 
//       length n with positive entries
//    theta: the rotation angle of Givens rotation. For details see Section 4 
//       of [1]
//    gamma: a parameter that controls the length of the optimal solution, 
//       see procedure OsGen3 in Section 6.0 in [1]
// 
//    b: right hand side 
//    A: m x n overdetermined matrix with rank n
//    xopt: the minimizer of tau*||x||_1 + 0.5||Ax-b||_2^2
//    fvalOpt: the value tau*||xopt||_1 + 0.5||Axopt-b||_2^2
#include<iostream>
#include<Eigen/Eigen>
#include<cmath>
#include<ctime>
#include<vector>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include <sys/time.h>

#define PI 3.14159265
// #define mydebug(x) std::cerr <<__LINE__ << ": " << std::string(x) << std::endl

using namespace std;
using namespace Eigen;


SparseMatrix<float>  A;
VectorXf b;
VectorXf xopt;
float ans;

int ls_gen_overdetermined(int n,int m,int k,int tau,
                            VectorXf s,float theta, float gamma ){
    int i,j,max,min;
    int idx[m];
    float ct=cos(theta);
    float st=sin(theta);
    Matrix2f  R;
    SparseMatrix<float> Gt(n,n),l_G(m,m),p(m,m);

    //givens rotation matrix 
    R << ct,-st,
        st,ct;
    Matrix2f Rt=R.transpose();
    //cout<<R<<'\n'<<Rt<<endl;

    //creat right singular vectors
    for(i=0;i<n;i=i+2){
        Gt.insert(i,i)=ct;
        Gt.insert(i,i+1)=st;
        Gt.insert(i+1,i)=-st;
        Gt.insert(i+1,i+1)=ct;
    }

    //Create left composition of Givens rotations
    for(i=0;i<m;i=i+2){
        l_G.insert(i,i)=ct;
        l_G.insert(i,i+1)=-st;
        l_G.insert(i+1,i)=st;
        l_G.insert(i+1,i+1)=ct;
    }

    //creat permutation matrix
    for(i=0;i<m;i++) idx[i]=i;
    for(i=m-1;i>=1;i--) swap(idx[i],idx[rand()%i]);
    for(i=0;i<m;i++) p.insert(i,idx[i])=1;

    //creat matrix A
    SparseMatrix<float> temp(m,n);
    for (i=0;i<n;i++){
        temp.insert(i,i)=s(i);

    }
    A = p*l_G*p*temp;

    //Create the optimal solution xopt.
    //Create noise
    struct  timeval  start;
    struct  timeval  end;
    unsigned long timer;
    gettimeofday(&start,NULL);
    
    VectorXf temp_s(n),xopt_temp(n),xopt_temp_abs(n),xopt_temp_max(n),xopt_temp_min(n);
    for(i=0;i<n;i++){
        temp_s(i)=gamma/(s(i)*s(i));
    }
    xopt_temp=Gt.transpose()*temp_s;
    xopt_temp_abs=xopt_temp.array().abs();
    //cout<<xopt_temp<<endl;
    xopt_temp_max=xopt_temp_abs;
    xopt_temp_min=xopt_temp_abs;
    xopt=VectorXf::Zero(n);
    std::ptrdiff_t max_i, min_j;
    VectorXf sub_g=VectorXf::Zero(n);    
    for(i=0;i<k/2;i++){
        xopt_temp_max.maxCoeff(&max_i);
        xopt(max_i)=xopt_temp_max(max_i);
        xopt_temp_max(max_i)=-1;
    }
    for(i=0;i<k/2;i++){
        xopt_temp_min.minCoeff(&min_j);
        xopt(min_j)=xopt_temp_min(min_j);
        xopt_temp_min(min_j)=10;
    }
    srand(time(0));     
    for(i=0;i<n;i++){
        if(xopt(i)==0){
            sub_g(i)=1.8*((rand()%10000)/10000.0)-0.9;
        }
        else if(xopt_temp(i)<0){ 
                xopt(i)=-xopt(i);
                sub_g(i)=-1;
        }
        else{
            sub_g(i)=1;
        }
    
    } 
    gettimeofday(&end,NULL);
    timer = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
    cout<<"time is "<<timer<<" us"<<endl;   //输出时间
    VectorXf e_temp=Gt * sub_g;
    for(i=0;i<n;i++){
        e_temp(i)=e_temp(i)/(s(i)*s(i));
    }
    VectorXf e=A*Gt.transpose()*e_temp;
    e=tau *e;

    //creat b;
    b=A*xopt+e;

    //calculate optimal objective function value
    ans=tau *xopt.lpNorm<1>()+0.5* e.squaredNorm();

    //  cout<<A<<endl;
    //  cout<<b<<endl;
    //  cout<<ans<<endl;
    //  cout<<xopt<<endl;
    return 0;
}

int main(){
    int n=8;
    int m=16;
    int k=2;
    int tau=1;
    float theta =1;
    float gamma =1;
    VectorXf s=VectorXf::Random(n);
    for(int i=0;i<n;i++){
        s(i)=abs(s(i))*20;
    }
    srand(time(0));
    cout<<"s: "<<'\n'<<s<<endl;
    ofstream  input( "input.txt",ios::app);
    ofstream  ansf( "ans.txt",ios::app);
    
    for(int i=0;i<10;i++){
        gamma=(rand()%10000)/10000.0;
        ls_gen_overdetermined(n, m,k, tau,s,theta,gamma);
    }
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            input<<A.coeffRef(i,j)<<' ';
        }
    }
    input <<b<<endl<<xopt<<endl;
    ansf<<ans<<endl;
}
