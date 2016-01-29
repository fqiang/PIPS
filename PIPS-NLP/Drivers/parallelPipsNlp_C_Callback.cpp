/* PIPS-NLP                                                         	*
 * Authors: Feng Qiang                    		*
 * (C) 2016 Argonne National Laboratory			*/

#include <cstdlib>

#include "parallelPipsNlp_C_Callback.h"

#include "../Core/NlpInfo/StructJuMPsInfo.h"
#include "../../Input/StructJuMPInput.h"
#include "../Core/NlpStoch/NlpPIPSIpmInterface.h"
#include "../Core/NlpStoch/sFactoryAug.h"
#include "../Core/NlpSolvers/FilterIPMStochSolver.h"


#include "NlpGenData.h"
#include "NlpGenVars.h"
#include "NlpGenResiduals.h"
#include "NlpInfoCallBack.h"


#include "FilterIPMSolver.h"


#include "NlpGenSparseWithSolver.h"
#include "cNlpGenSparseNLP.h"

#include "Status.h"

#include "pipsOptions.h"

#include "../par_macro.h"

extern "C"
PipsNlpProblemStructPtr CreatePipsNlpProblemStruct(
	MPI_Comm comm,
	int nnode,
	int n, int m,
	str_init_x0_cb init_x0,
	str_prob_info_cb prob_info,
	str_eval_f_cb eval_f,
	str_eval_g_cb eval_g,
	str_eval_grad_f_cb eval_grad_f,
	str_eval_jac_g_cb eval_jac_g,
	str_eval_h_cb eval_h,
	UserDataPtr userdata)
{
	MPI_Comm_rank(comm, &gmyid);
	MPI_Comm_size(comm, &gnprocs);
	std::cout << "on proc ["<<gmyid<<"] of ["<< gnprocs << "] MPI processes." <<std::endl;
	PAR_DEBUG("CreatePipsNlpProblemStruct - C");

	pipsOptions *pipsOpt = new pipsOptions();
	pipsOpt->readFile();
	pipsOpt->defGloOpt();

	PipsNlpProblemStructPtr retval = new PipsNlpProblemStruct;

	retval->comm = comm;
	retval->nnodes = nnode;
	retval->n = n;
	retval->m = m;
	retval->init_x0 = init_x0;
	retval->prob_info = prob_info;
	retval->eval_f = eval_f;
	retval->eval_g = eval_g;
	retval->eval_grad_f = eval_grad_f;
	retval->eval_jac_g = eval_jac_g;
	retval->eval_h = eval_h;
	retval->userdata = userdata;

	return retval;
}


extern int gInnerSCsolve;
extern int gNP_Alg;
extern int gAddSlackParallelSetting;
extern int gSymLinearSolver;
extern int gUseReducedSpace;

extern "C"
int PipsNlpSolveStruct( PipsNlpProblemStruct* prob)
{
	PAR_DEBUG("PipsNlpSolveStruct  - "<<gnprocs);
	MPI_Comm comm = prob->comm;

	pipsOptions *pipsOpt = new pipsOptions();
	pipsOpt->readFile();
	pipsOpt->defGloOpt();
	gInnerSCsolve=0;

	StructJuMPInput *s = new StructJuMPInput(prob);
	PAR_DEBUG("comm is "<<comm);
	assert(comm == MPI_COMM_WORLD);

	PAR_DEBUG("before PIPSIpmInterface created .." );
	NlpPIPSIpmInterface<sFactoryAug, FilterIPMStochSolver, StructJuMPsInfo> pipsIpm(*s,comm);
	PAR_DEBUG("PIPSIpmInterface created .." );

	//  delete s;
	PAR_DEBUG("AMPL NL  deleted ... solving");

	if (gmyid == 0) {
		std::cout << "  \n  -----------------------------------------------\n"
		<< "  NLP Solver \n"
		<< "  Argonne National Laboratory, 2016\n"
		<< "  -----------------------------------------------\n" <<std::endl;

		if(gUseReducedSpace>0)
		std::cout << "\n  Reduced Space Solver ------	 Reduced Space Solver with Umfpack and following linear solver.\n";

		if(0==gSymLinearSolver)
		std::cout << "\n  Linear system solver ------	 Ma27.\n\n";
		else if(1==gSymLinearSolver)
		std::cout << "\n  Linear system solver ------	 Ma57.\n\n";
		else if(2==gSymLinearSolver)
		std::cout << "\n  Linear system solver ------	 Pardiso.\n\n";
		else if(3==gSymLinearSolver)
		std::cout << "\n  Linear system solver ------	 Umfpack.\n\n";
	}

	pipsIpm.go();

	delete pipsOpt;

	return 0;
}


extern "C"
void FreePipsNlpProblemStruct(PipsNlpProblemStruct* prob){

}
