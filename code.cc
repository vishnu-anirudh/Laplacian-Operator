#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <sstream>
#include <cstdlib>
#include <fstream> 
#include <typeinfo>


// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include<Eigen/Dense>
#include<Eigen/SVD>
#include<Eigen/QR>

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>


//Using boost to generate values following gaussian distribution
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#define PI 3.14159265
// ----------------------------------------------------------------------------
using namespace OpenMesh;
using namespace Eigen;

using namespace std;
// ----------------------------------------------------------------------------

struct MyTraits : public OpenMesh::DefaultTraits 
{ 
   typedef OpenMesh::Vec3d Point; // use double-values points
   VertexAttributes( OpenMesh::Attributes::Normal | 
                   OpenMesh::Attributes::Color ); 
   FaceAttributes( OpenMesh::Attributes::Normal |
   		   OpenMesh::Attributes::Color ); 
}; 

typedef TriMesh_ArrayKernelT<MyTraits>  TriMesh;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double



//################# Adding noise function BEGINS #################// 
TriMesh add_noise(TriMesh myMesh1)
{//This function was primarily writted to check laplacian smoothing while writing the code initially. This was mainly used with the 3D sphere. However, this is not used in the code while taking the final output
   double zero_mean_noise;
      
  //Generating zero mean gaussian 
  boost::mt19937 rng; // Seed
  boost::normal_distribution<> nd(0.0, 1.0);
  boost::variate_generator<boost::mt19937&, 
                           boost::normal_distribution<> > var_nor(rng, nd);

   
for (TriMesh::VertexIter v_it=myMesh1.vertices_begin(); v_it!=myMesh1.vertices_end(); ++v_it)
{
   zero_mean_noise=var_nor()*0.01;
  myMesh1.point(v_it)[0]=myMesh1.point(v_it)[0]+zero_mean_noise;
  myMesh1.point(v_it)[1]=myMesh1.point(v_it)[1]+zero_mean_noise;
  myMesh1.point(v_it)[2]=myMesh1.point(v_it)[2]+zero_mean_noise;
 
}

return myMesh1;
}
//################# Adding noise function ENDS #################//




//################# Curvature function for beltrami BEGINS #################//
int curvature_beltrami(TriMesh beltrami_mesh, TriMesh myMesh, int option)
{
  cout<<"ENTERING CURVATURE CALCULATION (for laplace beltrami) "<<endl;
  IO::Options writeOptions;
  writeOptions+=IO::Options::VertexColor;
   
  TriMesh::Color blue(0,0,255);//Defining blue color
  TriMesh::Color red(255,0,0);//Defining red color
  TriMesh::Color green(0,255,0);//Defining green color
  
  TriMesh min_curv_mesh=myMesh;//Mesh for color-coding minimum curvature
  TriMesh max_curv_mesh=myMesh;//Mesh for color-coding maximum curvature
  
    for (TriMesh::VertexIter v_it=beltrami_mesh.vertices_begin(); v_it!=beltrami_mesh.vertices_end(); ++v_it)
    {

      TriMesh::Color max_vtx_color(0,0,0);
  TriMesh::Color min_vtx_color(0,0,0);
  int checker=0;
//----------------- CALCULATING GAUSS CURVATURE -----------------//
   TriMesh::VHandle p=v_it.handle();//Present vertex
  TriMesh::VHandle pt1, pt2; // Other two vertices of face
  TriMesh::Point   v1, v2,v3;   // Edges along the Face, coming from p
  TriMesh::Scalar   s;        // Stores the vector along the bisect of the triangle
  TriMesh::Scalar   T1=0.0;    // Theta
  TriMesh::Scalar   Tt =0.0;  // Total Theta
  TriMesh::Scalar   A =0.0,A1=0.0;  // Area 
  TriMesh::Scalar   At1=0.0;  // Total Area 
  TriMesh::Scalar   K =0.0;  // Gaussian curvature
  TriMesh::Scalar   kappa1, kappa2, kapp;  // Gaussian curvature
 // TriMesh::Scalar   cos_angle1 =0.0;  // Angle
   for (TriMesh::VertexFaceIter face = beltrami_mesh.vf_iter(v_it); face; ++face)
     {
     // calculate the angle of the face in the point. For that,
     // first acquire the points in the triangle
      TriMesh::FaceVertexIter verts = beltrami_mesh.fv_iter(face);
      if (p == verts.handle())
       {
            pt1 = (++verts).handle();
            pt2 = (++verts).handle();
        }
        else
        {
            pt1 = verts.handle();
            pt2 = (p == (++verts).handle()) ? (++verts).handle() : verts.handle();
        }
       
     // Calculate the edges
        v1    =    beltrami_mesh.point(p) - beltrami_mesh.point(pt1); 
        v2    =    beltrami_mesh.point(p) - beltrami_mesh.point(pt2); 
        v3 = ( beltrami_mesh.point(pt1) - beltrami_mesh.point(pt2) );
        s   =   ( v1.length() + v2.length() + v3.length())/2;
    
        A1=sqrt( s*(s-v1.length()) * (s-v2.length()) * (s-v3.length()) );
        v1.normalize(); v2.normalize();
        
     // Calculate the angle  
        T1 = acos(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
        
        Tt+=T1;
        At1+=A1;
             
     }
     OpenMesh::Vec3f omVx1 = OpenMesh::vector_cast<Vec3f>(beltrami_mesh.point(v_it));
  Eigen::Map< Eigen::Vector3f > H(omVx1.data());//Mapping normal of laplacian-operator applied point(to eigen vector)
  
     H=H/2;//Mean curvature
     Tt=2*PI-Tt;
     At1=At1*0.333;
     K=Tt/At1;//Gauusian curvature 

 if((pow(H.norm(),2)-K) >0)
      {
      
     	kappa1=H.norm()+sqrt(pow(H.norm(),2)-K);//Kappa1
     	kappa2=H.norm()-sqrt(pow(H.norm(),2)-K);//Kappa2
     	checker=1;//Signifying that kappa1 and kappa2 exist
      }
      else
      {
               kapp=sqrt(abs(pow(H.norm(),2)-K));
             	checker=-1;//Signifying that kappa1 and kappa2 DO NOT exist
      }
  
  
//-------------------- Curvature color -----------------//
 
 kappa2=log(abs(kappa2));
 kappa1=(kappa1);
    if(checker==1)//Checking if kappa exists
   {
    if(abs(kappa1)*255<255)//First level is blue
    {
      max_vtx_color[0]=abs(kappa1)*blue[0];
      max_vtx_color[1]=abs(kappa1)*blue[1];
      max_vtx_color[2]=abs(kappa1)*blue[2];
      max_curv_mesh.set_color(v_it,max_vtx_color);
     } 
    else if(abs(kappa1)*255<2*255)//Next level is green
    {
      max_vtx_color[0]=(abs(kappa1)-1)*green[0];
      max_vtx_color[1]=(abs(kappa1)-1)*green[1];
      max_vtx_color[2]=(abs(kappa1)-1)*green[2];
      max_curv_mesh.set_color(v_it,max_vtx_color);
    }
    else if(abs(kappa1)*255<=3*255)//Highest level is red
    {
      max_vtx_color[0]=(abs(kappa1)-2)*red[0];
      max_vtx_color[1]=(abs(kappa1)-2)*red[1];
      max_vtx_color[2]=(abs(kappa1)-2)*red[2];
      max_curv_mesh.set_color(v_it,max_vtx_color);
    }
  

    if(abs(kappa2)*255<1*255)
    {
      min_vtx_color[0]=abs(kappa2)*blue[0];
      min_vtx_color[1]=abs(kappa2)*blue[1];
      min_vtx_color[2]=abs(kappa2)*blue[2];
      min_curv_mesh.set_color(v_it,min_vtx_color);
    }
    else if(abs(kappa2)*255<2*255)
    {
      min_vtx_color[0]=(abs(kappa2)-1)*green[0];
      min_vtx_color[1]=(abs(kappa2)-1)*green[1];
      min_vtx_color[2]=(abs(kappa2)-1)*green[2];
      min_curv_mesh.set_color(v_it,min_vtx_color);
    }
    else if(abs(kappa2)*255<=3*255)
    {
      min_vtx_color[0]=(abs(kappa2)-2)*red[0];
      min_vtx_color[1]=(abs(kappa2)-2)*red[1];
      min_vtx_color[2]=(abs(kappa2)-2)*red[2];
      min_curv_mesh.set_color(v_it,min_vtx_color);
    }

  } 
  else if(checker==-1)//If the principal curvature cannot be calculated
  {
      max_vtx_color[0]=255;
      max_vtx_color[1]=255;
      max_vtx_color[2]=255;
      max_curv_mesh.set_color(v_it,max_vtx_color);

      min_vtx_color[0]=255;
      min_vtx_color[1]=255;
      min_vtx_color[2]=255;
      min_curv_mesh.set_color(v_it,min_vtx_color);
  }
   
  }
  
  if(option==1)
  {//Laplace Beltrami
   OpenMesh::IO::write_mesh(max_curv_mesh, "max_curvature_beltrami.ply",writeOptions);
  OpenMesh::IO::write_mesh(min_curv_mesh, "min_curvature_beltrami.ply",writeOptions);
  }
  else if (option==2)
  {//Explicit Laplace Beltrami
  OpenMesh::IO::write_mesh(max_curv_mesh, "max_curvature_explicit.ply",writeOptions);
  OpenMesh::IO::write_mesh(min_curv_mesh, "min_curvature_explicit.ply",writeOptions);
  }
   else if (option==3)
  {//Implicit Laplace Beltrami
  OpenMesh::IO::write_mesh(max_curv_mesh, "max_curvature_implicit.ply",writeOptions);
  OpenMesh::IO::write_mesh(min_curv_mesh, "min_curvature_implicit.ply",writeOptions);
  }
  cout<<"QUITTING CURVATURE CALCULATION (for laplace beltrami) "<<endl;
  return 0;
}  
  
//################# Curvature function for laplace beltrami ENDS #################






//################# Curvature for uniform laplace function BEGINS #################
int curvature_uniform(TriMesh laplac_mesh, TriMesh myMesh)
{
 cout<<"ENTERED CURVATURE CALCULATION (UNIFORM LAPLACE) FUNCTION"<<endl;
  IO::Options writeOptions;
  writeOptions+=IO::Options::VertexColor;
   
  TriMesh::Color blue(0,0,255);// blue color
  TriMesh::Color red(255,0,0);// red color
  TriMesh::Color green(0,255,0);// green color
  
//----------------------calculating MEAN CURVATURE--------------//
  TriMesh curv_mesh=laplac_mesh;
  TriMesh min_curv_mesh=myMesh;//Mesh for color-coding minimum curvature
  TriMesh max_curv_mesh=myMesh;//Mesh for color-coding maximum curvature

  Vector3f H;
  double cos_angle;
  
  for (TriMesh::VertexIter v_it=laplac_mesh.vertices_begin(); v_it!=laplac_mesh.vertices_end(); ++v_it)
{ 

  TriMesh::Color max_vtx_color(0,0,0);
  TriMesh::Color min_vtx_color(0,0,0);
  int checker=0;//to check whether principal curvature can be calculated or not
 
  OpenMesh::Vec3f omVx1 = OpenMesh::vector_cast<Vec3f>(laplac_mesh.normal(v_it));
  Eigen::Map< Eigen::Vector3f > eigen_normal_map(omVx1.data());//Mapping normal of laplacian-operator applied point(to eigen vector)
  
  OpenMesh::Vec3f omVx2 = OpenMesh::vector_cast<Vec3f>(laplac_mesh.point(v_it));
  Eigen::Map< Eigen::Vector3f > eigen_point_map(omVx2.data());//Mapping the result of laplacian-operator
  
   cos_angle=eigen_normal_map.dot(eigen_point_map)/(eigen_normal_map.norm()*eigen_point_map.norm());//cos_angle between the normal and the vector
   H=eigen_point_map/(eigen_normal_map.norm()*cos_angle);//Mean curvature
  

//------------------CALCULATING GAUSS CURVATURE------------------------//
   TriMesh::VHandle p=v_it.handle();//Present vertex
  TriMesh::VHandle pt1, pt2; // Other two vertices of face
  TriMesh::Point   v1, v2,v3;   // Edges along the Face, coming from p
  TriMesh::Scalar   s;        // Stores the vector along the bisect of the triangle
  TriMesh::Scalar   T1=0.0;    // Theta
  TriMesh::Scalar   Tt =0.0;  // Total Theta
  TriMesh::Scalar   A =0.0,A1=0.0;  // Area 
  TriMesh::Scalar   At1=0.0;  // Total Area 
  TriMesh::Scalar   K =0.0;  // Gaussian curvature
  TriMesh::Scalar   kappa1, kappa2, kapp;  // Gaussian curvature

   for (TriMesh::VertexFaceIter face = laplac_mesh.vf_iter(v_it); face; ++face)
     {
     // calculate the angle of the face in the point. For that,
     // first acquire the points in the triangle
      TriMesh::FaceVertexIter verts = laplac_mesh.fv_iter(face);
      if (p == verts.handle())
       {
            pt1 = (++verts).handle();//second point of the triangle 
            pt2 = (++verts).handle();//thrid point of the triangle
        }
        else
        {
            pt1 = verts.handle();
            pt2 = (p == (++verts).handle()) ? (++verts).handle() : verts.handle();
        }
       
     // Calculate the edges
        v1    =    laplac_mesh.point(p) - laplac_mesh.point(pt1); 
        v2    =    laplac_mesh.point(p) - laplac_mesh.point(pt2); 
        v3 = ( laplac_mesh.point(pt1) - laplac_mesh.point(pt2) );
        s   =   ( v1.length() + v2.length() + v3.length())/2;
    
        A1=sqrt( s*(s-v1.length()) * (s-v2.length()) * (s-v3.length()) );
        v1.normalize(); v2.normalize();        
     // Calculate the angle  
        T1 = acos(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);

        Tt+=T1;//Adding angles
        At1+=A1;//Adding area
             
     }
     Tt=2*PI-Tt; //2*Pi - sum-of-angles
     At1=At1*0.333;
    // K=Tt/At1; //(not scaling with respect to the area deficit)
     K=Tt; //Gaussian Curvature
      if((pow(H.norm(),2)-K) >0)
      {
     	kappa1=H.norm()+sqrt(pow(H.norm(),2)-K);//Kappa1
     	kappa2=H.norm()-sqrt(pow(H.norm(),2)-K);//Kappa2
     	checker=1;//Signifying that kappa1 and kappa2 exist
      }
      else
      {
        kapp=sqrt(abs(pow(H.norm(),2)-K));
        checker=-1;//Signifying that kappa1 and kappa2 DO NOT exist
      }
  
  
//-------------------Curvature color-----------------------//
 
   kappa2=log(abs(kappa2));//Taking the log here as the values of kappa2 are very small
   kappa1=(kappa1);
   if(checker==1)//Checking if kappa exists
   {
     if(abs(kappa1)*255<255)//Blue is the first level
     {
       max_vtx_color[0]=abs(kappa1)*blue[0];
       max_vtx_color[1]=abs(kappa1)*blue[1];
       max_vtx_color[2]=abs(kappa1)*blue[2];
       max_curv_mesh.set_color(v_it,max_vtx_color);
      } 
      else if(abs(kappa1)*255<2*255)//Green is the second level
      {
       max_vtx_color[0]=(abs(kappa1)-1)*green[0];
       max_vtx_color[1]=(abs(kappa1)-1)*green[1];
       max_vtx_color[2]=(abs(kappa1)-1)*green[2];
       max_curv_mesh.set_color(v_it,max_vtx_color);
      }
      else if(abs(kappa1)*255<=3*255)//Red is the highest level
      {
       max_vtx_color[0]=(abs(kappa1)-2)*red[0];
       max_vtx_color[1]=(abs(kappa1)-2)*red[1];
       max_vtx_color[2]=(abs(kappa1)-2)*red[2];
       max_curv_mesh.set_color(v_it,max_vtx_color);
      }
  

      if(abs(kappa2)*255<1*255)
      {
       min_vtx_color[0]=abs(kappa2)*blue[0];
       min_vtx_color[1]=abs(kappa2)*blue[1];
       min_vtx_color[2]=abs(kappa2)*blue[2];
       min_curv_mesh.set_color(v_it,min_vtx_color);
       //cout<<"Kappa2: 1: "<<kappa2<<endl;
       }
       else if(abs(kappa2)*255<2*255)
       {
        min_vtx_color[0]=(abs(kappa2)-1)*green[0];
        min_vtx_color[1]=(abs(kappa2)-1)*green[1];
        min_vtx_color[2]=(abs(kappa2)-1)*green[2];
        min_curv_mesh.set_color(v_it,min_vtx_color);
       // cout<<"Kappa2: 2: "<<kappa2<<endl;
        }  
        else if(abs(kappa2)*255<=3*255)
        {
          min_vtx_color[0]=(abs(kappa2)-2)*red[0];
          min_vtx_color[1]=(abs(kappa2)-2)*red[1];
          min_vtx_color[2]=(abs(kappa2)-2)*red[2];
          min_curv_mesh.set_color(v_it,min_vtx_color);
          //cout<<"Kappa2: 3: "<<kappa2<<endl;
         }
      } 
      else if(checker==-1)//if H^2-K<0
      {
       max_vtx_color[0]=255;
       max_vtx_color[1]=255;
       max_vtx_color[2]=255;
       max_curv_mesh.set_color(v_it,max_vtx_color);

       min_vtx_color[0]=255;
       min_vtx_color[1]=255;
       min_vtx_color[2]=255;
       min_curv_mesh.set_color(v_it,min_vtx_color);
     }
  
 
  }
  
  OpenMesh::IO::write_mesh(max_curv_mesh, "max_curvature_uniform.ply",writeOptions);
  OpenMesh::IO::write_mesh(min_curv_mesh, "min_curvature_uniform.ply",writeOptions);
  
  cout<<"QUITTING CURVATURE CALCULATION (UNIFORM LAPLACE) FUNCTION"<<endl;
  return 0;
}

//################# Curvature function for uniform laplace ENDS ###################//






//################# Laplace Beltrami function-1 BEGINS #################//
TriMesh laplace_beltrami1(TriMesh myMesh1)
{
cout<<"ENTERED LAPLACE-BELTRAMI FUNCTION"<<endl;
TriMesh beltrami_mesh=myMesh1;
 for (TriMesh::VertexIter v_it=myMesh1.vertices_sbegin(); v_it!=(myMesh1.vertices_end()); ++v_it)
{ 
  TriMesh::Point   prev_pt, next_pt, present_pt ;   
  OpenMesh::Vec3f v1_beta, v2_beta,v1_alpha,v2_alpha;
  TriMesh::Point lapl_bel(0,0,0);

  TriMesh::Scalar s=0.0;
  TriMesh::Scalar alpha=0.0, beta=0.0; //Angles
  TriMesh::Scalar cotan_sum;
  TriMesh::Scalar total_area=0.0;
  TriMesh::Scalar area=0.0;
  int is_inside=0;//See if the loop below is entered
//  TriMesh::Scalar cos_angle=0.0;
  for (TriMesh::VertexVertexIter vv_it=myMesh1.vv_iter(v_it); vv_it; ++vv_it)
  {
  TriMesh::VertexFaceIter face = myMesh1.vf_iter(vv_it);
  if(face.is_valid()){
  
     is_inside=1;  
     TriMesh::VertexVertexIter vv_it_prev=vv_it;
     TriMesh::VertexVertexIter vv_it_next=vv_it;
     prev_pt=myMesh1.point(--vv_it_prev);//Previous point to the present-one ring neighbour
     next_pt=myMesh1.point(++vv_it_next);//Next point to the present-one ring neighbour
     present_pt=myMesh1.point(vv_it);
    
    //Determining alpha 
     v1_alpha= myMesh1.point(v_it)-prev_pt;
     v2_alpha= present_pt-prev_pt; 
     v1_alpha.normalize(); v2_alpha.normalize(); 
     alpha = acos(v1_alpha[0] * v2_alpha[0] + v1_alpha[1] * v2_alpha[1] + v1_alpha[2] * v2_alpha[2]);//Angle Alpha
    
    //Determining beta 
     v1_beta= myMesh1.point(v_it)-next_pt;
     v2_beta= present_pt-next_pt;  
     
     area = 0.5 * v1_beta.norm() * v2_beta.norm();//area= (1/2)*a*b
     
     v1_beta.normalize(); v2_beta.normalize(); 
     beta = acos(v1_beta[0] * v2_beta[0] + v1_beta[1] * v2_beta[1] + v1_beta[2] * v2_beta[2]);//Angle Beta
     
     area=area*sin(beta);//now area=(1/2)*a*b * sin(angle-between-a-and-b)
     total_area+=area;//Total area    
     cotan_sum=tan((PI/2)-alpha)+tan((PI/2)-beta);//cot(aplha)+cot(beta)
     lapl_bel+=( (cotan_sum) * (  present_pt -myMesh1.point(v_it) ) );
     
     }   	
  }
  
  if(is_inside==1){
  total_area=0.333*total_area;
  beltrami_mesh.point(v_it)=myMesh1.point(v_it)+ ( 0.00000009*(lapl_bel/(2*total_area)) );// Laplace beltrami operator
  }

} //for-loop -->VertexIter v_it
    
cout<<"QUITTING LAPLACE BELTRAMI FUNCTION"<<endl;
return beltrami_mesh;
}//End of function
//################# Laplace Beltrami function-1 ENDS #################//




//################# Uniform-Laplace function BEGINS ###################//
TriMesh uniform_laplace(TriMesh myMesh1)
{
  cout<<"ENTERED UNIFORM LAPLACE FUNCTION"<<endl;
  TriMesh laplace_mesh1;
  laplace_mesh1=myMesh1;
  laplace_mesh1.request_vertex_colors();
for (TriMesh::VertexIter v_it=myMesh1.vertices_begin(); v_it!=myMesh1.vertices_end(); ++v_it)
{ 
  int count=0;
  TriMesh::Point lap_pt(0,0,0);
  for (TriMesh::VertexVertexIter vv_it=myMesh1.vv_iter(v_it); vv_it; ++vv_it)
  {
   	count+=1;
  	lap_pt+=myMesh1.point(vv_it)-myMesh1.point(v_it);//Subtracting from one ring neighbours	
  }
  lap_pt/=count;//Dividing by number of one-ring neighbours
  laplace_mesh1.point(v_it)=myMesh1.point(v_it)+0.9*lap_pt;//Uniform laplace operator
}
cout<<"QUITTING UNIFORM LAPLACE FUNCTION"<<endl;
return laplace_mesh1;
}
//################# Uniform-Laplace function ends #################//




//################# Diffusion Flow function BEGINS ####################//
TriMesh diffusion_flow_explicit(TriMesh myMesh1, int integration_opt)
{
  cout<<"ENTERED DIFFUSION FLOW FUNCTION"<<endl;
  TriMesh explicit_mesh1;
  explicit_mesh1=myMesh1;
  explicit_mesh1.request_vertex_colors();
  int num_vertices=0;//Total number of vertices
  int ind=0;
  std::map<TriMesh::VertexHandle, int> vertMap;//map between vertex handles and vertices
  
for (TriMesh::VertexIter v_it=myMesh1.vertices_sbegin(); v_it!=myMesh1.vertices_end(); ++v_it)
{ 
vertMap[v_it.handle()]=ind;//Indexing the vertices
  ++ind;
}//for-loop VertexIter v_it 
num_vertices=ind;
Eigen::MatrixXd mat_P(ind,3);//P matrix
Eigen::MatrixXd mat_Pnew(ind,3);//new P matrix
Eigen::SparseMatrix<double> sparse_matrix_L(ind,ind);//L sparse matrix
Eigen::SparseMatrix<double> sparse_matrix_M(ind,ind);//L sparse matrix
Eigen::SparseMatrix<double> sparse_matrix_A(ind,ind);//A sparse matrix--for linear system
Eigen::SparseMatrix<double> sparse_matrix_identity(ind,ind);//A sparse matrix--for linear system
Eigen::MatrixXd iden_mat(ind, ind);

ind=0;
for (TriMesh::VertexIter v_it=explicit_mesh1.vertices_sbegin(); v_it!=(explicit_mesh1.vertices_end()); ++v_it)
{  
  mat_P(ind,0)=explicit_mesh1.point(v_it)[0];
  mat_P(ind,1)=explicit_mesh1.point(v_it)[1];
  mat_P(ind,2)=explicit_mesh1.point(v_it)[2];
  ++ind;
  
  TriMesh::Point   prev_pt, next_pt, present_pt ;   
  OpenMesh::Vec3f v1_beta, v2_beta,v1_alpha,v2_alpha,v3;
  TriMesh::Scalar lapl_bel=0;

  TriMesh::Scalar s=0.0;
  TriMesh::Scalar alpha=0.0, beta=0.0; //Angles
  TriMesh::Scalar cotan_sum;
  TriMesh::Scalar total_area=0.0;//Sum of area of triangles formed by one ring neighbours
  TriMesh::Scalar area=0.0;//Area of a triangle
  int is_inside=0;//See if the code enters the loop
  
  for (TriMesh::VertexVertexIter vv_it=explicit_mesh1.vv_iter(v_it); vv_it; ++vv_it)
  {
  TriMesh::Scalar lapl_bel_point=0;
  TriMesh::VertexFaceIter face = explicit_mesh1.vf_iter(vv_it);
  if(face.is_valid()){
  
     is_inside=1;  
     TriMesh::VertexVertexIter vv_it_prev=vv_it;
     TriMesh::VertexVertexIter vv_it_next=vv_it;
     prev_pt=explicit_mesh1.point(--vv_it_prev);//Previous point to the one-ring neighbour connected point
     next_pt=explicit_mesh1.point(++vv_it_next);//Next point to the one-ring neighbour connected point
     present_pt=explicit_mesh1.point(vv_it);
    
    //Determining alpha
     v1_alpha= explicit_mesh1.point(v_it)-prev_pt;
     v2_alpha= present_pt-prev_pt; 
     v3=explicit_mesh1.point(v_it)-present_pt;
     
     //Another method of calculating area
     //s=( v1_alpha.norm()+v2_alpha.norm()+v3.norm() )/2;
     //area=sqrt(s* (s-v1_alpha.norm()) * (s-v2_alpha.norm()) * (s-v3.norm()) );
     
     v1_alpha.normalize(); v2_alpha.normalize(); 
     alpha = acos(v1_alpha[0] * v2_alpha[0] + v1_alpha[1] * v2_alpha[1] + v1_alpha[2] * v2_alpha[2]);//Angle Alpha
    
    //Determining beta 
     v1_beta= explicit_mesh1.point(v_it)-next_pt;
     v2_beta= present_pt-next_pt;  
     area = 0.5 * v1_beta.norm() * v2_beta.norm();//area= (1/2)*a*b
     v1_beta.normalize(); v2_beta.normalize(); 
     beta = acos(v1_beta[0] * v2_beta[0] + v1_beta[1] * v2_beta[1] + v1_beta[2] * v2_beta[2]);//Angle Beta
     area=area*sin(beta);//now area=(1/2)*a*b * sin(angle-between-a-and-b) 
     total_area+=area;//Total area
         
     cotan_sum=tan((PI/2)-alpha)+tan((PI/2)-beta);//cot(aplha)+cot(beta)
     lapl_bel_point=( (cotan_sum) );//Sum of the cotangents 
     lapl_bel+=lapl_bel_point;//Adding the contangent-sum
     sparse_matrix_L.insert(vertMap[v_it.handle()],vertMap[vv_it.handle()])=lapl_bel_point;
     sparse_matrix_M.insert(vertMap[v_it.handle()],vertMap[vv_it.handle()])=lapl_bel_point;
     }//if-statement
   }//for-loop VertexVertexIter(vv_it)
   total_area=0.333*total_area;
     
     if(total_area<pow(10,-20)){
     total_area=pow(10,-20);
     }
     
     lapl_bel=(-1)*lapl_bel;
     
     sparse_matrix_M.insert(vertMap[v_it.handle()],vertMap[v_it.handle()])=lapl_bel;//M Matrix (used to implicit integration)
     
     lapl_bel=(lapl_bel/(2*total_area));//Dividing by 2*area when i=j (thus, includes the D matrix)
     
    sparse_matrix_L.insert(vertMap[v_it.handle()],vertMap[v_it.handle()])=lapl_bel;//L Matrix
   
}//for-loop VertexIter (v_it)
//cout<<"P "<<mat_P<<endl;

if (integration_opt==1)
{//Forward Euler integration
cout<<"Forward Euler Integration"<<endl;
float lamda=pow(10,-15);
mat_Pnew=mat_P+lamda*(sparse_matrix_L * mat_P);
}
else if (integration_opt==2)
{
cout<<"Backward Euler Integration"<<endl;
ConjugateGradient<SparseMatrix<double> > solver; //Solve Ax=b using conjugate gradient

//Eigen::SimplicialCholesky<SpMat> solver(sparse_matrix_A);//Using sparse cholesky
float lamda1=pow(10,-10);
//sparse_matrix_A=sparse_matrix_identity-  lamda1*sparse_matrix_L;
sparse_matrix_A=sparse_matrix_M-  lamda1*sparse_matrix_M*sparse_matrix_L;
solver.compute(sparse_matrix_A);
cout<<"after setting the solver"<<endl;
mat_P=sparse_matrix_M*mat_P;
mat_Pnew.col(0) = solver.solve(mat_P.col(0));
mat_Pnew.col(1) = solver.solve(mat_P.col(1));
mat_Pnew.col(2) = solver.solve(mat_P.col(2));
cout<<"solved the linear system!"<<endl;

if(solver.info()!=Success) {
  // decomposition failed
  cout<<"SOLVER NOT WORKING!!"<<endl;
}

}
ind=0;
for (TriMesh::VertexIter v_it=explicit_mesh1.vertices_sbegin(); v_it!=(explicit_mesh1.vertices_end()); ++v_it)
{ 
 explicit_mesh1.point(v_it)[0]=mat_Pnew(ind,0);
 explicit_mesh1.point(v_it)[1]=mat_Pnew(ind,1);
 explicit_mesh1.point(v_it)[2]=mat_Pnew(ind,2);
 ++ind;
}

cout<<"QUITTING DIFFUSION FLOW FUNCTION"<<endl;
return explicit_mesh1;
}//function end

//################# Diffusion Flow function ENDS ########################//





//#################### MAIN() function BEGINS #################### //
int main(int argc, char *argv[])
{
  
  TriMesh myMesh1;  

  IO::Options readOptions,writeOptions;
  readOptions+=IO::Options::VertexColor;
  readOptions+=IO::Options::VertexNormal;
  writeOptions+=IO::Options::VertexColor;

  IO::read_mesh(myMesh1, "/home/venkateshvishnuanirudh/bunny/reconstruction/bun_zipper_res2.ply",readOptions);//Loading bunny
  //IO::read_mesh(myMesh1, "/home/venkateshvishnuanirudh/spheres/ico_sphere4.ply",readOptions);//Loading sphere
  
  myMesh1.request_vertex_colors();

 // myMesh1=add_noise(myMesh1);//Adding uniform noise
  
  cout<<" "<<endl;
  
  //Uniform Laplace
  TriMesh laplace_mesh=uniform_laplace(myMesh1);//Applying the uniform laplace operator
  laplace_mesh.update_normals();
  myMesh1.update_normals();  
  cout<<" "<<endl;
  int l=curvature_uniform(laplace_mesh, myMesh1);
  
  cout<<" "<<endl;
  
 
  //Laplace Beltrami
  TriMesh beltrami_mesh=laplace_beltrami1(myMesh1);
  cout<<" "<<endl;
  int l1=curvature_beltrami(beltrami_mesh, myMesh1,1);//option=1
  cout<<" "<<endl;
 
  
  //Forward Euler (explicit integration)
  cout<<"Forward Euler Integration "<<endl;
  TriMesh explicit_intermediate=myMesh1;
  TriMesh explicit_mesh;
  int num_iteration1=5;
  for(int iter=0;iter<num_iteration1;++iter){
  cout<<"Iteration: "<<iter<<endl;
  explicit_mesh=diffusion_flow_explicit(explicit_intermediate,1);//integration_option=1 for explicit
  explicit_intermediate=explicit_mesh;
  }
  int l2=curvature_beltrami(explicit_mesh, myMesh1,2);//option=2 for saving explicit-related curvatur
  
  cout<<" "<<endl;
  cout<<"Backward Euler Integration "<<endl;
 //Backward Euler (explicit integration)
  TriMesh implicit_intermediate=myMesh1;
  TriMesh implicit_mesh;
  int num_iteration2=5;
  for(int iter=0;iter<num_iteration2;++iter){
  cout<<"Iteration: "<<iter<<endl;
  implicit_mesh=diffusion_flow_explicit(implicit_intermediate,2);//integration_option=1 for explicit
  implicit_intermediate=implicit_mesh;
  }
  int l3=curvature_beltrami(implicit_mesh, myMesh1,3);//option=3 for saving implicit-related curvatures
  
  cout<<" "<<endl;

  //OpenMesh::IO::write_mesh(myMesh1, "Input.ply");//Storing the output
  
  return 0;
}
//#################### MAIN() function ENDS #################### //

//********************** END OF FILE **************************//
