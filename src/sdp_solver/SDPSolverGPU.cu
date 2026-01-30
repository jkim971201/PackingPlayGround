#include <cassert>
#include <unordered_map>
#include <limits>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cuda_linalg/CudaUtil.h"
#include "cuda_linalg/CudaVectorAlgebra.h"
#include "cuda_linalg/CudaMatrixAlgebra.h"
#include "cuda_linalg/CudaSparseMatrixAlgebra.h"

#include "SDPInstance.h"
#include "SDPSolverGPU.h"

__global__ void slack_projection_kernel(
  const int     num_dual,
  const int     num_eq_constraint,
  const int     num_ineq_constraint,
  const double  rho,
  const double* b,
  const double* ARRT,
  const double* lambda,
        double* y) 
{
  const int constraint_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(constraint_id < num_dual)
  {
    if(constraint_id < num_eq_constraint) // Equality Constraint
      y[constraint_id] = b[constraint_id];
    else if(constraint_id >= num_eq_constraint && constraint_id < num_dual) // Inequality Constraint
    {
      double h_i = b[constraint_id];
      double ARRT_i = ARRT[constraint_id];
      double lambda_i = lambda[constraint_id];
      y[constraint_id] = max(ARRT_i + lambda_i / rho, h_i);
    }
    else
      assert(0);
  }
}

__global__ void constraint_computation_kernel(
  const int     num_row,
  const int     num_eq_constraint,
  const int     num_ineq_constraint,
  const int*    eq_constraint_index,
  const int*    ineq_constraint_index,
  const double* weight,
        double* output)
{
  const int num_constraint = num_eq_constraint + num_ineq_constraint;
  const int constraint_id = blockIdx.x * blockDim.x + threadIdx.x;

  if(constraint_id < num_eq_constraint)
  {
    int eq_id = constraint_id;
    int flatten_index = eq_constraint_index[eq_id];
    double weight_val = weight[constraint_id] * 2.0;
    atomicAdd(&(output[flatten_index]), weight_val);
  }
  else if(constraint_id >= num_eq_constraint && constraint_id < num_constraint)
  {
    int ineq_id = constraint_id - num_eq_constraint;
    int flatten_index = ineq_constraint_index[ineq_id];
    // id1 = row, id2 = col
    int id1 = flatten_index % num_row;
    int id2 = flatten_index / num_row;
    assert(id1 < id2);
    double weight_val = weight[constraint_id] * 2.0;

    // column-major flatten -> row + num_row * col
    int index1 = id1 + num_row * id1; // (id1, id1) -> +1
    int index2 = id1 + num_row * id2; // (id1, id2) -> -1
    int index3 = id2 + num_row * id1; // (id2, id1) -> -1
    int index4 = id2 + num_row * id2; // (id2, id2) -> +1

    atomicAdd(&(output[index1]), +weight_val);
    atomicAdd(&(output[index2]), -weight_val);
    atomicAdd(&(output[index3]), -weight_val);
    atomicAdd(&(output[index4]), +weight_val);
  }
}

namespace sdp_solver
{

struct LbfgsNode
{
  double alpha;
  double beta; // rho in LBFGS Wiki
  CudaFlattenMatrix<double> s;
  CudaFlattenMatrix<double> y;
  std::shared_ptr<LbfgsNode> prev;
  std::shared_ptr<LbfgsNode> next;

  LbfgsNode(int n, int k)
    : prev(nullptr), next(nullptr)
  {
    s.initialize(n, k);
    y.initialize(n, k);
    alpha = 0.0;
    beta  = 0.0;
  }
};

void convertEigenToCudaFlattenMatrix(
  const EigenSMatrix& mat_eigen,
  CudaFlattenMatrix<double>& mat_cuda)
{
	const int num_row = mat_eigen.rows();
	const int num_col = mat_eigen.cols();

  mat_cuda.initialize(num_row, num_col);

  std::vector<double> h_data(num_row * num_col, 0.0);
  
  const int outer_size = mat_eigen.outerSize();
  for(int k = 0; k < outer_size; k++)
  {
    for(EigenSMatrix::InnerIterator it(mat_eigen, k); it; ++it)
    {
      int row = it.row();
      int col = it.col();
      double val = static_cast<double>(it.value());
      // NOTE: cuBLAS is Column-Major (IMPORTANT!!!)
      h_data[row + col * num_col] = val;
    }
  }

  auto& d_data = mat_cuda.getFlattenVector();

  thrust::copy(h_data.begin(), h_data.end(), 
               d_data.begin());
}

SDPSolverGPU::SDPSolverGPU(std::shared_ptr<SDPInstance> sdp_inst)
  : n_                   (0),
    m_                   (0),
    p_                   (0),
    primal_obj_          (0.0),
    primal_infeasibility_(0.0),
    b_l2_norm_           (0.0),
    b_inf_norm_          (0.0),
    C_inf_norm_          (0.0),
    runtime_             (0.0),
    is_maximization_     (false)
{
  int num_eq_constr = sdp_inst->eq_const_val.size();
  int num_ineq_constr = sdp_inst->ineq_const_val.size();

  setObjective(sdp_inst->obj_matrix);

  auto& eq_const_matrix   = sdp_inst->eq_const_matrix;
  auto& eq_const_val      = sdp_inst->eq_const_val;

  auto& ineq_const_matrix = sdp_inst->ineq_const_matrix;
  auto& ineq_const_val    = sdp_inst->ineq_const_val;

  for(int i = 0; i < num_eq_constr; i++)
    addEqualityConstraint(eq_const_matrix[i], eq_const_val[i]);

  for(int i = 0; i < num_ineq_constr; i++)
    addInequalityConstraint(ineq_const_matrix[i], ineq_const_val[i]);

  initParams();
}

void
SDPSolverGPU::makeSparseMatrixM()
{
  EigenSMatrix sparseM(m_ + p_, n_ * n_);
  EigenSMatrix sparseMT(n_ * n_, m_ + p_);

  std::vector<int> eq_const_index;
  std::vector<int> ineq_const_index;

  for(int i = 0; i < m_; i++)
  {
    const EigenSMatrix& A_eigen = A_[i];
    const int outer_size = A_eigen.outerSize();
    for(int k = 0; k < outer_size; k++)
    {
      for(EigenSMatrix::InnerIterator it(A_eigen, k); it; ++it)
      {
        int row = it.row();
        int col = it.col();
        double val = static_cast<double>(it.value());

        // cublas is column-major!!
        sparseM.coeffRef(i, row + col * n_) = val;

        // For compressed matrix
        eq_const_index.push_back(row + col * n_);
      }
    }
  }

  for(int i = 0; i < p_; i++)
  {
    const EigenSMatrix& G_eigen = G_[i];
    const int outer_size = G_eigen.outerSize();
    for(int k = 0; k < outer_size; k++)
    {
      for(EigenSMatrix::InnerIterator it(G_eigen, k); it; ++it)
      {
        int row = it.row();
        int col = it.col();
        double val = static_cast<double>(it.value());

        // cublas is column-major!!
        sparseM.coeffRef(i + m_, row + col * n_) = val;

        // For compressed matrix
        if(row < col)
          ineq_const_index.push_back(row + col * n_);
      }
    }
  }

  assert(eq_const_index.size() == m_);
  assert(ineq_const_index.size() == p_);

  d_eq_const_index_.resize(m_);
  d_ineq_const_index_.resize(p_);

  thrust::copy(eq_const_index.begin(), eq_const_index.end(),
               d_eq_const_index_.begin());

  thrust::copy(ineq_const_index.begin(), ineq_const_index.end(),
               d_ineq_const_index_.begin());

  sparseMT = sparseM.transpose();

  d_M_.initialize(sparseM);
  d_MT_.initialize(sparseMT);
}

void
SDPSolverGPU::loadDataOnGPU()
{
  auto t_load1 = std::chrono::high_resolution_clock::now();

  const int num_dual = m_ + p_;

  // Step 1. Load Objective Matrix
  convertEigenToCudaFlattenMatrix(C_, d_C_);

  // Step 2. Load Equality Constraint and Inequality Constraint 
  d_b_.resize(num_dual);
  thrust::copy(b_.begin(), b_.end(), d_b_.begin());
  thrust::copy(h_.begin(), h_.end(), d_b_.begin() + m_);

  // Step 3. Prepare R matrix (primal solution)
  d_R_.initialize(n_, target_rank_);
  d_Rnew_.initialize(n_, target_rank_);
  d_RGrad_.initialize(n_, target_rank_);
  d_RGradnew_.initialize(n_, target_rank_);

  // Step 4. Prepare etc.
  d_LbfgsDirection_.initialize(n_, target_rank_);

  d_LbfgsWorkspace_.initialize(n_, target_rank_);

  d_UVT_workspace_.initialize(n_, n_);
  d_X_.initialize(n_, n_);

  d_grad_workspace_matrix_.initialize(n_, n_);
  d_grad_workspace_vector_.resize(num_dual);

  d_linesearch_workspace_.resize(num_dual);

  d_infeasibility_workspace_.resize(num_dual);

  // Step 5. Prepare lambda vector (dual solution)
  d_lambda_.resize(num_dual);

  // Step 6. Prepare ARRT vector (constraint value)
  d_ARRT_.resize(num_dual);
  d_ARDT_.resize(num_dual);
  d_ADDT_.resize(num_dual);

  // Step 7. Prepare A matrices
//  d_A_.resize(num_dual);
//  for(int i = 0; i < m_; i++)
//  {
//    const EigenSMatrix& h_Ai = A_[i];
//    CudaFlattenMatrix<double>& d_Ai = d_A_[i];
//    convertEigenToCudaFlattenMatrix(h_Ai, d_Ai);
//  }
//
//  for(int i = 0; i < p_; i++)
//  {
//    const EigenSMatrix& h_Gi = G_[i];
//    CudaFlattenMatrix<double>& d_Gi = d_A_[i + m_];
//    convertEigenToCudaFlattenMatrix(h_Gi, d_Gi);
//  }

  // Step 8. Prepare LBFGS nodes
  lbfgs_head_ = std::make_shared<LbfgsNode>(n_, target_rank_);
  auto node = lbfgs_head_;
  for(int i = 0; i < param_.lbfgs_length - 1; i++)
  {
    // this loop goes backward, i.e. the latest (head) to the oldest.
    std::shared_ptr<LbfgsNode> next_node       
      = std::make_shared<LbfgsNode>(n_, target_rank_);
    node->next = next_node;
    next_node->prev = node;
    node = node->next;

    if(i == param_.lbfgs_length - 2)
    {
      node->next = lbfgs_head_;
      lbfgs_head_->prev = node;
    }
  }

  // Step 9. Prepare y vector
  d_y_.resize(num_dual);
  thrust::copy(d_b_.begin(), d_b_.end(), d_y_.begin());

  // Step 10. M Matrix
  makeSparseMatrixM();
  
  auto t_load2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_load_gpu  = t_load2 - t_load1;
  double time_load_gpu_double = time_load_gpu.count();

  if(param_.verbose == true)
    printf("    Load data on GPU takes %5.2f seconds\n", time_load_gpu_double);
}

void
SDPSolverGPU::initializeRho()
{
  int num_constr = m_ + p_;
  double num_constr_double = static_cast<double>(num_constr);
  //rho_ = 1.0 / std::sqrt(num_constr_double);
  //rho_ = 1.0 * std::sqrt(num_constr_double);
  rho_ = 1.0 * std::sqrt(n_);
}

void
SDPSolverGPU::findTargetRank()
{
  int num_constr = m_ + p_;
  int optimal_rank 
    = static_cast<int>(std::sqrt(2 * num_constr) + 1.0);
  //target_rank_ = std::min(optimal_rank, n_);
  target_rank_ = std::sqrt(num_constr / 4);
  //target_rank_ = std::sqrt(num_constr);
}

void
SDPSolverGPU::computeUVT(
  const CudaFlattenMatrix<double>& U,
  const CudaFlattenMatrix<double>& V,
        CudaFlattenMatrix<double>& UVT) /* Output X = UV^T */
{
  symmetricRankKUpdate2(0.5, U, V, UVT);
  //symmetricRankKUpdate1(1.0, U, UVT);
}

void
SDPSolverGPU::computeConstraintValue(
  const CudaFlattenMatrix<double>& U,
  const CudaFlattenMatrix<double>& V,
        CudaVector<double>&        AUVT)
{
  computeUVT(U, V, d_UVT_workspace_);
  sparseMatrixVectorMult(d_M_, d_UVT_workspace_.getFlattenVector(), AUVT);
}

double
SDPSolverGPU::computeObjectiveValue(
  const CudaFlattenMatrix<double>& U,
  const CudaFlattenMatrix<double>& V)
{
  computeUVT(U, V, d_UVT_workspace_);
  return fmatrixInnerProduct(d_C_, d_UVT_workspace_);
}

void 
SDPSolverGPU::computeGradient(
  const double                     rho,
  const CudaVector<double>&        y,
  const CudaVector<double>&        lambda,
  const CudaVector<double>&        ARRT,
  const CudaFlattenMatrix<double>& R,
  const CudaFlattenMatrix<double>& C,
        CudaFlattenMatrix<double>& Grad)
{
  d_grad_workspace_matrix_.fillZero();

  // workspace_matrix = 2C
  fmatrixAxpy(2.0, C, d_grad_workspace_matrix_);

  // workspace_vector = 0
  d_grad_workspace_vector_.fillZero();

  // workspace_vector = rho * Vector( Tr(ARRT) )
  vectorAxpy(rho, ARRT, d_grad_workspace_vector_);

  // workspace_vector = rho * Vector( Tr(ARRT) ) - rho * y
  vectorAxpy(-rho, y, d_grad_workspace_vector_);

  // workspace_vector = rho * Vector( Tr(ARRT) ) - rho * y + lambda
  vectorAxpy(+1.0, lambda, d_grad_workspace_vector_);

  // workspace_matrix = 2C + 2 * sum(weight(i) * Ai)
  computeWeightedMatrixSumCustomKernel(d_grad_workspace_vector_, d_grad_workspace_matrix_);

  // Grad = workspace_matrix * R
  computeSymMatrixMult(1.0, d_grad_workspace_matrix_, R, Grad);
}

void
SDPSolverGPU::computeWeightedMatrixSum(
  const CudaVector<double>& weight,
  CudaFlattenMatrix<double>& grad_workspace_matrix)
{
  const int num_dual = m_ + p_;

  std::vector<double> h_weight(num_dual);
  thrust::copy(weight.begin(), weight.end(),
               h_weight.begin());

  for(int i = 0; i < d_A_.size(); i++)
  {
    const auto& Ai = d_A_[i];
    fmatrixAxpy(2.0 * h_weight[i], Ai, grad_workspace_matrix);
  }
}

void
SDPSolverGPU::computeWeightedMatrixSumCustomKernel(
  const CudaVector<double>& weight,
  CudaFlattenMatrix<double>& grad_workspace_matrix)
{
  const int num_row = n_;
  const int num_dual = m_ + p_;
  const int num_block = num_dual;
  const int num_thread = 16;

  constraint_computation_kernel<<<num_block, num_thread>>>(
    num_row,                     /* num_row of constraint matrices */
    m_,                          /* num_eq_constraint              */
    p_,                          /* num_ineq_constraint            */
    d_eq_const_index_.data(),    /* index of each equality constraint matrix   */
    d_ineq_const_index_.data(),  /* index of each inequality constraint matrix */
    weight.data(),               /* weight vector */
    grad_workspace_matrix.getFlattenVector().data() /* output matrix */
  );
}

void
SDPSolverGPU::computeLbfgsDirection(
  const int                        lbfgs_iter,
  const double                     rho,
  const CudaVector<double>&        b,
  const CudaVector<double>&        lambda,
  const CudaVector<double>&        ARRT,
  const CudaFlattenMatrix<double>& C,
  const CudaFlattenMatrix<double>& R,
  const CudaFlattenMatrix<double>& Grad,
        CudaFlattenMatrix<double>& D)
{
  D.fillZero();

  //printDenseMatrixRowMajor(Grad, "Lbfgs InputGrad");

  if(lbfgs_iter == 0)
  {
    // D = -Grad
    fmatrixAxpy(-1.0, Grad, D);
  }
  else
  {
    thrust::copy(Grad.getFlattenVector().begin(),
                 Grad.getFlattenVector().end(),
                 d_LbfgsWorkspace_.getFlattenVector().begin() );

    const int lbfgs_length = param_.lbfgs_length;
    int num_node = (lbfgs_iter < lbfgs_length) ? lbfgs_iter : lbfgs_length;

    // Start from the latest
    auto node = lbfgs_head_->prev; 
    for(int k = 0; k < num_node; ++k)
		{
      // New to Old
      double temp = fmatrixInnerProduct(node->s, d_LbfgsWorkspace_);
      node->alpha = node->beta * temp;
      fmatrixAxpy(-1.0 * (node->alpha), node->y, d_LbfgsWorkspace_);
      node = node->prev;
    }

    // Start from the oldest
    node = node->next; 
    for(int k = 0; k < num_node; ++k)
		{
      // Old to New
      double temp = node->alpha - node->beta * fmatrixInnerProduct(node->y, d_LbfgsWorkspace_);
      fmatrixAxpy(temp, node->s, d_LbfgsWorkspace_);
      node = node->next;
    }

    // D = -LbfgsWorkspace
    fmatrixAxpy(-1.0, d_LbfgsWorkspace_, D);
  }

  //printDenseMatrixRowMajor(D, "Lbfgs Direction");
}

inline double quarticFunction(double a, double b, double c, double d, double x)
{
  return a * x * x * x * x + b * x * x * x + c * x * x + d * x;
}

inline int solve_qubic(
  double a,  
  double b, 
  double c, 
  double d, 
  std::vector<double>& roots)
{
  int num_root = 0;
  double A = b * b - 3 * a * c;
  double B = b * c - 9 * a * d;
  double C = c * c - 3 * b * d;
  double delta = B * B - 4 * A * C; // Discriminant of Qubic Equation
  roots.resize(3, 0.0);
  if(A == 0 && B == 0)
  {
    roots[0] = std::max(roots[0], -c / b);
    num_root = 1;
  }
  else if(delta > 0)
  {
    double Y1   = A * b + 1.5 * a * (-B + std::sqrt(delta));
    double Y2   = A * b + 1.5 * a * (-B - std::sqrt(delta));
    double Y1_3 = Y1 >= 0.0 ? std::pow(Y1, 1.0 / 3.0) : -1.0 * std::pow(-Y1, 1.0 / 3.0);
    double Y2_3 = Y2 >= 0.0 ? std::pow(Y2, 1.0 / 3.0) : -1.0 * std::pow(-Y2, 1.0 / 3.0);
    roots[0] = std::max(roots[0], (-b - Y1_3 - Y2_3) / 3 / a);
    num_root = 1;
  }
  else if(delta == 0 && A != 0 && B != 0)
  {
    double K = B / A;
    roots[0] = -b / a + K;
    roots[1] = -K / 2;
    num_root = 2;
  }
  else if(delta < 0)
  {
    double sqA   = std::sqrt(A);
    double T     = (A*b - 1.5 * a * B) / ( A * sqA);
    double theta = std::acos(T);
    double csth  = std::cos(theta / 3);
    double sn3th = std::sqrt(3) * std::sin(theta / 3);
    double root1 = (-b - 2 * sqA * csth) / 3 / a;
    double root2 = (-b + sqA * (csth + sn3th) ) / 3 / a;
    double root3 = (-b + sqA * (csth - sn3th) ) / 3 / a;
    roots[0] = root1;
    roots[1] = root2;
    roots[2] = root3;
    num_root = 3;
  }
  return num_root;
}

double
SDPSolverGPU::lineSearchLbfgs(
  const double              rho,
  const double              CDDT,
  const double              CRDT,
  const CudaVector<double>& b_vec,
  const CudaVector<double>& ARRT,
  const CudaVector<double>& ARDT,
  const CudaVector<double>& ADDT,
  const CudaVector<double>& lambda)
{
  d_linesearch_workspace_.fillZero();
  vectorAxpy(+rho, ARRT,   d_linesearch_workspace_);
  vectorAxpy(-rho, b_vec,  d_linesearch_workspace_);
  vectorAxpy(+1.0, lambda, d_linesearch_workspace_);
  // linesearch_workspace = rho * ARRT - rho * b + lambda

  const double a = 0.5 * rho * innerProduct(ADDT, ADDT);
  const double b = 2.0 * rho * innerProduct(ARDT, ADDT);
  const double c = CDDT + innerProduct(ADDT, d_linesearch_workspace_) 
                 + 2.0 * rho * innerProduct(ARDT, ARDT);
  const double d = 2.0 * CRDT 
                 + 2.0 * innerProduct(ARDT, d_linesearch_workspace_);

  std::vector<double> roots(3, 0.0);
  int num_roots = solve_qubic(4 * a, 3 * b, 2 * c, d, roots);
  assert(num_roots > 0);

  const double stepsize_max = 1.0;

  double f_lb = 0.0;
  double f_ub = quarticFunction(a, b, c, d, stepsize_max);

  double f_root0 = std::numeric_limits<double>::max();
  double f_root1 = std::numeric_limits<double>::max();
  double f_root2 = std::numeric_limits<double>::max();

  double root0 = roots[0];
  double root1 = roots[1];
  double root2 = roots[2];

  if(num_roots >= 1)
  {
    if(root0 > 1e-20 && root0 <= stepsize_max)
      f_root0 = quarticFunction(a, b, c, d, root0);
  }
  if(num_roots >= 2)
  {
    if(root1 > 1e-20 && root1 <= stepsize_max)
      f_root1 = quarticFunction(a, b, c, d, root1);
  }
  if(num_roots >= 3)
  {
    if(root2 > 1e-20 && root2 <= stepsize_max)
      f_root2 = quarticFunction(a, b, c, d, root2);
  }

  double argmin_f = 0.0;
  double min_f = std::min(std::min(std::min(std::min(f_lb, f_ub), f_root0), f_root1), f_root2);
  if(std::abs(min_f - f_lb) < 1e-10)
    argmin_f = 0.0;
  if(std::abs(min_f - f_ub) < 1e-10)
    argmin_f = 1.0;
  if(std::abs(min_f - f_root0) < 1e-10)
    argmin_f = root0;
  if(std::abs(min_f - f_root1) < 1e-10)
    argmin_f = root1;
  if(std::abs(min_f - f_root2) < 1e-10)
    argmin_f = root2;

  // printf("[LineSearch] a : %f b : %f c : %f d : %f tau : %f\n", a, b, c, d, argmin_f);

  return argmin_f;
}

bool
SDPSolverGPU::checkLbfgsConvergence(
  const double                     rho,
  const CudaFlattenMatrix<double>& Grad)
{
  const double grad_l2_norm = compute2Norm(Grad.getFlattenVector());
   
  double rho_certificate_tol = param_.rho_certificate / rho;
  double rho_certificate_val = grad_l2_norm / (1.0 + C_inf_norm_);

  //printf("[DEBUG] grad_l2_norm        : %f\n", grad_l2_norm);
  //printf("[DEBUG] C_inf_norm_         : %f\n", C_inf_norm_);
  //printf("[DEBUG] rho_certificate_val : %f\n", rho_certificate_val);
  //printf("[DEBUG] rho_certificate_tol : %f\n", rho_certificate_tol);

  if(rho_certificate_val < rho_certificate_tol)
    return true;
  else 
    return false;
}

int
SDPSolverGPU::solveLbfgs(
  const double                     rho,
  const CudaVector<double>&        b,
  const CudaVector<double>&        lambda,
  const CudaFlattenMatrix<double>& C,
        CudaFlattenMatrix<double>& Rout)
{
  const int max_lbfgs_iter = param_.max_lbfgs_iter;

  auto& R    = d_R_;
  auto& Rnew = d_Rnew_;

  auto& Grad    = d_RGrad_;
  auto& Gradnew = d_RGradnew_;

  auto& ARRT    = d_ARRT_;

  Grad.fillZero();

  int lbfgs_iter = 0;

  computeConstraintValue(R, R, ARRT);

  computeGradient(rho, b, lambda, ARRT, R, C, Grad);

  for(; lbfgs_iter < max_lbfgs_iter; lbfgs_iter++)
  {
    /* Compute D */
    computeLbfgsDirection(
      lbfgs_iter, 
      rho,
      b,
      lambda,
      ARRT,
      C,
      R,
      Grad,
      d_LbfgsDirection_);

    /* Compute Stepsize tau */
    double CDDT = computeObjectiveValue(d_LbfgsDirection_, d_LbfgsDirection_);
    double CRDT = computeObjectiveValue(R,                 d_LbfgsDirection_);

    computeConstraintValue(R,                 d_LbfgsDirection_, d_ARDT_);
    computeConstraintValue(d_LbfgsDirection_, d_LbfgsDirection_, d_ADDT_);

    double lbfgs_stepsize 
      = lineSearchLbfgs(rho_, CDDT, CRDT, b, ARRT, d_ARDT_, d_ADDT_, lambda);

    //printf("LBFGS Step: %f\n", lbfgs_stepsize);

    /* Update Rnew = R + tau*D  */
    fmatrixAdd(1.0, lbfgs_stepsize, R, d_LbfgsDirection_, Rnew);
    computeConstraintValue(Rnew, Rnew, ARRT);

    //printDenseMatrixRowMajor(R, "Rnew");

    /* Compute Next Gradient */
    computeGradient(rho, b, lambda, ARRT, Rnew, C, Gradnew);

    /* Update the Latest LBFGS Node */
    auto& s = lbfgs_head_->s;
    auto& y = lbfgs_head_->y;

    fmatrixAdd(1.0, -1.0, Rnew,    R,    s);
    fmatrixAdd(1.0, -1.0, Gradnew, Grad, y);

    lbfgs_head_->beta = 1.0 / fmatrixInnerProduct(y, s);
    //printf("  Lbfgs beta : %f\n", lbfgs_head_->beta);

    lbfgs_head_= lbfgs_head_->next;

    Rnew.swap(R);
    Gradnew.swap(Grad);

    //printf("  Lbfgs Iter : %d\n", lbfgs_iter);
    bool lbfgs_convergence = checkLbfgsConvergence(rho_, Grad);
    if(lbfgs_convergence == true)
      break;
  }

  Rout.swap(R);

  return lbfgs_iter;
}

void
SDPSolverGPU::updateDual(
  double rho,
  const CudaVector<double>& y,
  const CudaVector<double>& ARRT,
        CudaVector<double>& lambda)
{
  vectorAxpy(+rho, ARRT, lambda);
  vectorAxpy(-rho,    y, lambda);
  // lambda <- lambda + rho * ARRT - rho * y
}

void
SDPSolverGPU::updateSlack(
  int num_eq_constraint,
  int num_ineq_constraint,
  double rho,
  const CudaVector<double>& b,
  const CudaVector<double>& ARRT,
  const CudaVector<double>& lambda,
        CudaVector<double>& y)
{
  const int num_dual = num_eq_constraint + num_ineq_constraint;
  const int num_block = num_dual;
  const int num_thread = 16;

  slack_projection_kernel<<<num_block, num_thread>>>(
    num_dual,
    num_eq_constraint,
    num_ineq_constraint,
    rho,
    b.data(),
    ARRT.data(),
    lambda.data(),
    y.data() );
}

void
SDPSolverGPU::setInitialR(CudaFlattenMatrix<double>& R)
{
  std::vector<double> h_R(n_ * target_rank_);

  double rand_val = 0.0;
  for(int i = 0; i < n_ * target_rank_; i++)
  {
    rand_val  = static_cast<double>(rand()) / RAND_MAX;
    rand_val -= static_cast<double>(rand()) / RAND_MAX;
    h_R[i]    = rand_val;
  }

  thrust::copy(h_R.begin(), h_R.end(),
               R.getFlattenVector().begin());

  // printDenseMatrixRowMajor(R, "Initial R");
}

void
SDPSolverGPU::precomputeNorm()
{
  b_l2_norm_  = compute2Norm(d_b_);
  b_inf_norm_ = computeVectorMax(d_b_);

  C_inf_norm_ = 0.0;
  const int outer_size = C_.outerSize();
  for(int k = 0; k < outer_size; k++)
  {
    for(EigenSMatrix::InnerIterator it(C_, k); it; ++it)
    {
      double val = std::abs(it.value());
      C_inf_norm_ = std::max(C_inf_norm_, val);
    }
  }
}

bool
SDPSolverGPU::checkAlmConvergence(const CudaVector<double>& ARRT)
{
  vectorAdd(1.0, -1.0, ARRT, d_y_, d_infeasibility_workspace_);

  double y_inf_norm = computeVectorMax(d_y_);

  primal_infeasibility_ = compute2Norm(d_infeasibility_workspace_) / (1 + y_inf_norm);

  if(primal_infeasibility_ < param_.tol_alm)
    return true;
  else
    return false;
}

void
SDPSolverGPU::solveALM()
{
  initializeRho();

  findTargetRank();

  if(param_.verbose == true)
    printStartInfo(target_rank_, rho_);

  // target_rank must be set first by findTargetRank()
  loadDataOnGPU();

  precomputeNorm();

  setInitialR(d_R_);

  const int max_alm_iter = param_.max_alm_iter;

  int alm_iter = 0;

  if(param_.verbose == true)
    printMetricRow();

  auto t_start = std::chrono::high_resolution_clock::now();
  for(; alm_iter < max_alm_iter; alm_iter++)
  {
    int lbfgs_iter = solveLbfgs(rho_, d_y_, d_lambda_, d_C_, d_R_);

    computeConstraintValue(d_R_, d_R_, d_ARRT_);

    updateSlack(m_, p_, rho_, d_b_, d_ARRT_, d_lambda_, d_y_);

    updateDual(rho_, d_y_, d_ARRT_, d_lambda_);

    bool alm_convergence = checkAlmConvergence(d_ARRT_);
    if(alm_convergence == true)
      break;

    rho_ *= param_.rho_update_factor;
    primal_obj_ = computeObjectiveValue(d_R_, d_R_);

    auto t_progress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime_progress = t_progress - t_start;
    double runtime_progress_double = runtime_progress.count();

    if(param_.verbose == true)
    {
      printProgress(alm_iter, 
                    lbfgs_iter, 
                    rho_, 
                    primal_obj_, 
                    primal_infeasibility_, 
                    runtime_progress_double);
    }
  }

  //printDenseMatrixRowMajor(d_R_, "FinalR");
  
  // Compute Final Solution : X = RR^T
  computeUVT(d_R_, d_R_, d_X_);

  //printDenseMatrixRowMajor(d_X_, "FinalX");

//  std::vector<double> h_const(d_ARRT_.size());
//  thrust::copy(d_ARRT_.begin(), d_ARRT_.end(), h_const.begin());
//
//  std::vector<double> h_b(d_b_.size());
//  thrust::copy(d_b_.begin(), d_b_.end(), h_b.begin());
//
//  std::vector<double> h_y(d_y_.size());
//  thrust::copy(d_y_.begin(), d_y_.end(), h_y.begin());
//
//  std::vector<double> h_R(d_R_.getFlattenVector().size());
//  thrust::copy(d_R_.getFlattenVector().begin(), d_R_.getFlattenVector().end(), h_R.begin());
//
//  std::unordered_map<int, std::pair<int, int>> index_to_pair;
//  int index = 0;
//  for(int i = 0; i < n_ - 2; i++)
//  {
//    for(int j = i + 1; j < n_ - 2; j++)
//      index_to_pair[index++] = {i, j};
//  }
//
//  for(int i = 0; i < p_; i++)
//  {
//    printf("[%2d] ARRT: %f val: %f y: %f", i, h_const[i + m_], h_b[i + m_], h_y[i + m_]);
//    if(h_const[i + m_] < h_b[i + m_])
//    {
//      auto [id1, id2] = index_to_pair[i];
//      double x1 = h_R[id1 + 2];
//      double y1 = h_R[id1 + 2 + n_];
//
//      double x2 = h_R[id2 + 2];
//      double y2 = h_R[id2 + 2 + n_];
//
//      double dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
//
//      printf(" Violation! (%d - %d) Dist: %f", id1, id2, dist);
//    }
//    printf("\n");
//  }
}

EigenDMatrix
SDPSolverGPU::solve()
{
  if(param_.verbose == true)
    printBanner();

  auto t1 = std::chrono::high_resolution_clock::now();

  solveALM();

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime = t2 - t1;
  runtime_ = runtime.count();

  std::vector<double> h_X_dense(n_ * n_);
  const auto& d_X_data = d_X_.getFlattenVector();

  thrust::copy(d_X_data.begin(), d_X_data.end(),
               h_X_dense.begin());

  EigenDMatrix dense_matrix(n_, n_);
  for(int i = 0; i < n_; i++)
  {
    for(int j = 0; j < n_; j++)
    {
      double val = h_X_dense[i * n_ + j];
      dense_matrix(i, j) = val;
    }
  }

  if(param_.verbose == true)
    printFinish();

  return dense_matrix;
}

void
SDPSolverGPU::initParams()
{
  param_.max_alm_iter      = 100; 
  param_.max_lbfgs_iter    = 500;
  param_.log_freq          = 1;  
  param_.lbfgs_length      = 2;
  param_.tol_alm           = 1e-4;
  param_.rho_certificate   = 0.1;
  param_.rho_update_factor = 1.05;
  param_.verbose           = true;
}

void
SDPSolverGPU::setObjective(EigenSMatrix& C)
{
  C_ = C;
  n_ = C_.rows();
}

void
SDPSolverGPU::setMaximization()
{
  C_ = -1 * C_;
  is_maximization_ = true;
}

void
SDPSolverGPU::setVerbose(bool is_verbose)
{
  param_.verbose = is_verbose;
}

void
SDPSolverGPU::addEqualityConstraint(EigenSMatrix& Ai, double bi)
{
  m_++;
  A_.emplace_back(Ai);
  b_.emplace_back(bi);
}

void
SDPSolverGPU::addInequalityConstraint(EigenSMatrix& Gi, double hi)
{
  p_++;
  G_.emplace_back(Gi);
  h_.emplace_back(hi);
}

void
SDPSolverGPU::print() const
{
  std::cout << "Objective" << std::endl;
  std::cout << C_ << std::endl;

  std::cout << "Equality Constraint" << std::endl;
  for(auto& Ai : A_)
    std::cout << Ai << std::endl;
  std::cout << "b vector" << std::endl;
  for(int i = 0; i < m_; i++)
    std::cout << b_[i] << " " << std::endl;
  std::cout << std::endl;

  std::cout << "Inequality Constraint" << std::endl;
  for(auto& Gi : G_)
    std::cout << Gi << std::endl;
  for(int i = 0; i < p_; i++)
    std::cout << h_[i] << " " << std::endl;
  std::cout << std::endl;
}

void
SDPSolverGPU::printBanner() const
{
  std::string problem = 
    is_maximization_ == true ? "Maximization" : "Minimization";
  printf("=================================================================\n");
  printf("    Problem                    : %12s\n", problem.c_str());
  printf("    Matrix Size                : %12d\n", n_);
  printf("    Num   Equality Constraints : %12d\n", m_);
  printf("    Num Inequality Constraints : %12d\n", p_);
}

void
SDPSolverGPU::printStartInfo(int target_rank, double initial_rho) const
{
  printf("    Target Rank (k)            : %12d    \n", target_rank);
  printf("    Initial Rho                : %12.2f  \n", initial_rho);
}

void
SDPSolverGPU::printMetricRow() const
{
  printf("-----------------------------------------------------------------\n");
  printf("| ITER | LBFGS |   RHO   |  PRIMAL_OBJ  | PRIMAL_INF | Time (s) |\n");
  printf("-----------------------------------------------------------------\n");
}

void
SDPSolverGPU::printFinish() const
{
  double primal_obj_display = 
    is_maximization_ == true ? -primal_obj_ : primal_obj_;
  printf("-----------------------------------------------------------------\n");
  printf("    Time Elapsed (s) : %10.5f\n", runtime_);
  printf("    Primal Objective : %10.5f\n", primal_obj_display);
  printf("-----------------------------------------------------------------\n");
  printf("=================================================================\n");
}

void
SDPSolverGPU::printProgress(
  int    alm_iter,
  int    lbfgs_iter,
  double rho,
  double primal_obj,
  double primal_inf,
  double runtime) const
{
  double primal_obj_display = 
    is_maximization_ == true ? -primal_obj : primal_obj;
  printf("| %4d | ",  alm_iter);
  printf("%5d | ",    lbfgs_iter);
  printf("%7.3f |",   rho);
  printf("%13.5f | ", primal_obj_display);
  printf("%10.5f | ",  primal_inf);
  printf("%8.4f | ",  runtime);
  printf("\n");

}


}
