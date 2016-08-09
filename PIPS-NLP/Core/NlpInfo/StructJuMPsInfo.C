

#include "StructJuMPsInfo.h"
#include "StructJuMPInput.h"

#include "sData.h"
#include "sVars.h"
#include "sTree.h"

#include "../../PIPS-NLP/global_var.h"
#include "../PIPS-NLP/Core/Utilities/PerfMetrics.h"

StructJuMPsInfo::StructJuMPsInfo()
{
	assert(false);
}
StructJuMPsInfo::StructJuMPsInfo(sData *data_in):sInfo(data_in)
{
	assert(false);
}

StructJuMPsInfo::StructJuMPsInfo(sData *data_in, stochasticInput& in)
	:sInfo(data_in)
{
	MESSAGE("StructJuMPsInfo ( data_in , stochasticInput)  - id "<< nodeId());
	parent = NULL;
	stochInput = &(dynamic_cast<StructJuMPInput&>(in));

	MESSAGE("  in StructJuMPsInfo constr comm"<<data_in->stochNode->commWrkrs<<" "<<mpiComm<<"  "<<MPI_COMM_NULL);

	//	iAmDistrib=0;
	//	  if( MPI_COMM_NULL!=mpiComm) {
	int size;
	MPI_Comm_size(mpiComm, &size);
	MESSAGE("size of parallel procs "<<size);
	//	    iAmDistrib = size==1?0:1;
	//	  }
	assert(MPI_COMM_NULL!=mpiComm);
	assert(size == gnprocs);
	createChildren(data_in,*stochInput);

//	data_in->inputNlp = this;
}

StructJuMPsInfo::StructJuMPsInfo(sData *data_in, stochasticInput& in, const int idx)
	:sInfo(data_in)
{
	MESSAGE("StructJuMPsInfo ( data_in , structJuMPInput, "<<idx<<") id ("<<nodeId()<<")");

	stochInput = &(dynamic_cast<StructJuMPInput&>(in));
//	data_in->inputNlp = this;

//	iAmDistrib = 0;
//	if (MPI_COMM_NULL != mpiComm) {
		int size;
		MPI_Comm_size(mpiComm, &size);
//		iAmDistrib = size == 1 ? 0 : 1;
//	}
	MESSAGE("number of children "<<data_in->children.size());
	createChildren(data_in,*stochInput);
}


StructJuMPsInfo::~StructJuMPsInfo()
{

}

int StructJuMPsInfo::nodeId()
{
	return stochNode->id();
}

void StructJuMPsInfo::createChildren(sData *data_in, stochasticInput& in){
	MESSAGE("createChildren");
//	int mype_;
//		MPI_Comm_rank(in.prob->comm/* MPI_COMM_WORLD*/, &mype_);

	for (size_t it = 0; it < data_in->children.size(); it++) {
		if (stochNode->children[it]->commWrkrs != MPI_COMM_NULL) {
			AddChild(new StructJuMPsInfo(data_in->children[it], in, it));
		}
		else {
			MESSAGE("comm null "<<MPI_COMM_NULL<<" commwrk "<<stochNode->children[it]->commWrkrs);
			AddChild(new sInfoDummy());
		}
		children[it]->parent = this;
		//	this->iAmDistrib = 0;
	}
}

double StructJuMPsInfo::ObjValue(NlpGenVars * vars){
	MESSAGE("enter ObjValue -");
	sVars* svars = dynamic_cast<sVars*>(vars);
	StochVector& vars_X = dynamic_cast<StochVector&>(*svars->x);
	OoqpVector& local_X = *(dynamic_cast<StochVector&>(*svars->x).vec);

	std::vector<double> local_var(locNx,0);
	local_X.copyIntoArray(&local_var[0]);

	double robj = 0.0;
	if(parent==NULL)
	{
		MESSAGE(" ObjValue - parent is NULL");
		double objv = 0.0;
		if(gmyid == 0)
		{
			MESSAGE("ObjValue - gmyid=="<<gmyid);
			assert(nodeId() == 0);
			CallBackData cbd = {stochInput->prob->userdata,nodeId(),nodeId(),0};
			double obj;
#ifdef NLPTIMING
			double stime = MPI_Wtime();
#endif
			stochInput->prob->eval_f(&local_var[0],&local_var[0],&obj,&cbd);
#ifdef NLPTIMING
			gprof.t_model_evaluation += MPI_Wtime()-stime;
			gprof.n_feval += 1;
#endif
			objv += obj;
			PRINT_ARRAY("local_var",&local_var[0],locNx);
			MESSAGE("objv = "<<objv);
		}
		for(size_t it=0;it<children.size();it++) {
			MESSAGE("it - "<<it );
			objv += children[it]->ObjValue(svars->children[it]);
		}
		MESSAGE("objv = "<<objv);
		MPI_Allreduce(&objv, &robj, 1, MPI_DOUBLE, MPI_SUM, mpiComm);
		MESSAGE("ObjValue - after reduce - global robj="<<robj);
	}
	else
	{
		MESSAGE("ObjValue - parent  not NULL - "<<nodeId());
		int parid = parent->stochNode->id();
		assert(parid == 0);
		std::vector<double> parent_var(parent->locNx,0);
		OoqpVector* parent_X = (vars_X.parent->vec);
		parent_X->copyIntoArray(&parent_var[0]);
		assert(nodeId() != 0);
		CallBackData cbd = {stochInput->prob->userdata,nodeId(),nodeId(),0};
#ifdef NLPTIMING
		double stime = MPI_Wtime();
#endif
		stochInput->prob->eval_f(&parent_var[0],&local_var[0],&robj,&cbd);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime()-stime;
		gprof.n_feval += 1;
#endif
		robj = robj;
		PRINT_ARRAY("parent_var",&parent_var[0],parent->locNx);
		PRINT_ARRAY("local_var",&local_var[0], locNx);
		MESSAGE("robj="<<robj);
	}

	MESSAGE("end ObjValue "<<robj);
	return robj;
}

int StructJuMPsInfo::ObjGrad(NlpGenVars * vars, OoqpVector *grad){
	MESSAGE("enter ObjGrad");
	sVars * svars = dynamic_cast<sVars*>(vars);
	StochVector& vars_X = dynamic_cast<StochVector&>(*svars->x);
	OoqpVector& local_X = *(dynamic_cast<StochVector&>(*svars->x).vec);
	StochVector* sGrad = dynamic_cast<StochVector*>(grad);

	assert(parent == NULL);
	assert(nodeId()==0);

	std::vector<double> local_var(locNx,0);
	local_X.copyIntoArray(&local_var[0]);

	std::vector<double> local_grad(locNx,0.0);
//	MESSAGE("gymyid ="<<gmyid);
	if(gmyid == 0)
	{
	  CallBackData cbd = {stochInput->prob->userdata, nodeId(), nodeId(),0};
#ifdef NLPTIMING
		double stime = MPI_Wtime();
#endif
		stochInput->prob->eval_grad_f(&local_var[0],&local_var[0],&local_grad[0],&cbd);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime() - stime;
		gprof.n_grad_f += 1;
#endif
		PRINT_ARRAY("local_var",&local_var[0],locNx);
		PRINT_ARRAY("local_grad",local_grad,locNx);
	}

	for(size_t it=0; it<children.size(); it++){
		MESSAGE("it - "<<it);
		(children[it])->ObjGrad_FromSon(svars->children[it],sGrad->children[it], &local_grad[0]);
	}
	PRINT_ARRAY("local_grad",local_grad,locNx);

	std::vector<double> rgrad(locNx,0);
	MPI_Allreduce(&local_grad[0], &rgrad[0], locNx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	sGrad->vec->copyFromArray(&rgrad[0]);
	PRINT_ARRAY("after reduce - rgrad", &rgrad[0], locNx);
	MESSAGE("exit ObjGrad ");
	return 1;
}

void StructJuMPsInfo::ObjGrad_FromSon(NlpGenVars* vars, OoqpVector* grad, double* pgrad)
{
	MESSAGE("enter ObjGrad_FromSon - "<<nodeId());
	assert(parent!=NULL);
//	assert(parent->locNx == pgrad.size());
	sVars * svars = dynamic_cast<sVars*>(vars);
	StochVector& vars_X = dynamic_cast<StochVector&>(*svars->x);
	OoqpVector& local_X = *(dynamic_cast<StochVector&>(*svars->x).vec);

	std::vector<double> local_var(locNx,0);
	local_X.copyIntoArray(&local_var[0]);

	assert(parent->stochNode->id()==0);
	double parent_var[parent->locNx];
	OoqpVector* parent_X = (vars_X.parent->vec);
	parent_X->copyIntoArray(&parent_var[0]);


	PRINT_ARRAY("parent_var",parent_var,parent->locNx);
	PRINT_ARRAY("local_var",&local_var[0],locNx);
	std::vector<double> parent_part(parent->locNx,0.0);
	CallBackData cbd_parent = {stochInput->prob->userdata, nodeId(), parent->stochNode->id(),0};
#ifdef NLPTIMING
	double stime = MPI_Wtime();
#endif
	stochInput->prob->eval_grad_f(parent_var,&local_var[0],&parent_part[0],&cbd_parent);
#ifdef NLPTIMING
	gprof.t_model_evaluation += MPI_Wtime() - stime;
	gprof.n_grad_f += 1;
#endif

	MESSAGE(" --- parent contribution -");
	PRINT_ARRAY("parent_part",parent_part,parent->locNx);

	std::vector<double> this_part(locNx,0.0);
	CallBackData cbd_this = {stochInput->prob->userdata, nodeId(), nodeId(),0};
#ifdef NLPTIMING
	stime = MPI_Wtime();
#endif
	stochInput->prob->eval_grad_f(&parent_var[0],&local_var[0],&this_part[0],&cbd_this);
#ifdef NLPTIMING
	gprof.t_model_evaluation += MPI_Wtime() - stime;
	gprof.n_grad_f += 1;
#endif

	MESSAGE(" --- this node -");
	PRINT_ARRAY("this_part",this_part,locNx);

	StochVector* sGrad = dynamic_cast<StochVector*>(grad);
	sGrad->vec->copyFromArray(&this_part[0]);
	for(int i = 0;i<parent->locNx;i++)
		pgrad[i] += parent_part[i];
}


void StructJuMPsInfo::ConstraintBody(NlpGenVars * vars, OoqpVector *conEq,OoqpVector *conIneq){
	MESSAGE("enter ConstraintBody  - nodeid - "<<nodeId());
//	MESSAGE("S1 -- ----");
	sVars * svars = dynamic_cast<sVars*>(vars);
	StochVector& vars_X = dynamic_cast<StochVector&>(*svars->x);
	OoqpVector& local_X = *(dynamic_cast<StochVector&>(*svars->x).vec);
//	MESSAGE("S2 -- ----");

	std::vector<double> local_var(locNx,0.0);
	local_X.copyIntoArray(&local_var[0]);
	IF_VERBOSE_DO( local_X.print(); );
	PRINT_ARRAY("local_var ",&local_var[0], locNx);


	std::vector<double> coneq(locMy,0);
	std::vector<double> coninq(locMz,0);
	if(parent!=NULL) {
		assert(parent->stochNode->id()==0);
		double parent_var[parent->locNx];
		OoqpVector* parent_X = (vars_X.parent->vec);
		parent_X->copyIntoArray(&parent_var[0]);
		CallBackData cbd = {stochInput->prob->userdata,nodeId(),nodeId(),0};
#ifdef NLPTIMING
		double stime = MPI_Wtime();
#endif
		stochInput->prob->eval_g(&parent_var[0],&local_var[0],&coneq[0],&coninq[0],&cbd);

#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime()-stime;
		gprof.n_eval_g += 1;
#endif
	}
	else
	{
		assert(nodeId()==0);
		CallBackData cbd = {stochInput->prob->userdata,nodeId(),nodeId(),0};
#ifdef NLPTIMING
		double stime = MPI_Wtime();
#endif
		stochInput->prob->eval_g(&local_var[0],&local_var[0],&coneq[0],&coninq[0],&cbd);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime()-stime;
		gprof.n_eval_g += 1;
#endif
		int e_ml = stochInput->nLinkECons();
		int i_ml = stochInput->nLinkICons();
		if (e_ml > 0)
		{
		    SimpleVector linkeq(e_ml);
		    linkeq.setToZero();
		    for(size_t it=0; it<children.size(); it++){
		        //int m,n;
			//children[it]->Emat->getSize( m, n );
		        children[it]->Emult(1.0, linkeq, 1.0, *(dynamic_cast<StochVector&>(*svars->children[it]->x).vec));
		    }
		    double* buffer = new double[e_ml];
		    MPI_Allreduce(linkeq.elements(), buffer, e_ml, MPI_DOUBLE, MPI_SUM, mpiComm);
       		    for(int i=0; i<e_ml; i++)
		    {
		        linkeq.elements()[i] = buffer[i];
		    }  
		    Emult(1.0, linkeq, 1.0, *(dynamic_cast<StochVector&>(*svars->x).vec));
		    for(int i=0; i<e_ml; i++)
		    {
		        coneq[i+locMy-e_ml] = linkeq.elements()[i];
		    }
		    delete[] buffer;
		}
                if (i_ml > 0)
		{
                    SimpleVector linkinq(i_ml);
                    linkinq.setToZero();
                    for(size_t it=0; it<children.size(); it++){
		      children[it]->Fmult(1.0, linkinq, 1.0, *(dynamic_cast<StochVector&>(*svars->children[it]->x).vec));
		    }		  
		    double* buffer = new double[i_ml];
                    MPI_Allreduce(linkinq.elements(), buffer, i_ml, MPI_DOUBLE, MPI_SUM, mpiComm);
                    for(int i=0; i<i_ml; i++)
		    {
                        linkinq.elements()[i] = buffer[i];
		    }
                    Fmult(1.0, linkinq, 1.0, *(dynamic_cast<StochVector&>(*svars->x).vec));
                    for(int i=0; i<i_ml; i++)
		    {
			coninq[i+locMz -i_ml] = linkinq.elements()[i];
		    }
                    delete[] buffer;
		}
	}

	StochVector* sconeq = dynamic_cast<StochVector*>(conEq);
//	MESSAGE("sconeq size "<<sconeq->vec->n);
	assert(sconeq->vec->n == locMy);
	sconeq->vec->copyFromArray(&coneq[0]);

	StochVector* sconinq = dynamic_cast<StochVector*>(conIneq);
//	MESSAGE("sconinq size "<<sconinq->vec->n);
	assert(sconinq->vec->n == locMz);
	sconinq->vec->copyFromArray(&coninq[0]);

	for(size_t it=0; it<children.size(); it++){
		MESSAGE("it - "<<it);
		(children[it])->ConstraintBody(svars->children[it],sconeq->children[it],sconinq->children[it]);
	}
	MESSAGE("end ConstraintBody");
}

void StructJuMPsInfo::JacFull(NlpGenVars* vars, GenMatrix* JacA, GenMatrix* JaC)
{
//	note: no linking constraint handling
	MESSAGE("enter JacFull - "<<nodeId());

	long long mA, nA, mC, nC, mB,nB,mD,nD;
	Amat->getSize(mA,nA);
	Cmat->getSize(mC,nC);
	Bmat->getSize(mB,nB);
	Dmat->getSize(mD,nD);
//	MESSAGE(" Amat "<<mA<<"  "<<nA<<"  nz"<<Amat->numberOfNonZeros());
//	MESSAGE(" Cmat "<<mC<<"  "<<nC<<"  nz"<<Cmat->numberOfNonZeros());
//	MESSAGE(" Bmat "<<mB<<"  "<<nB<<"  nz"<<Bmat->numberOfNonZeros());
//	MESSAGE(" Dmat "<<mD<<"  "<<nD<<"  nz"<<Dmat->numberOfNonZeros());

	sVars * svars = dynamic_cast<sVars*>(vars);
	StochVector& vars_X = dynamic_cast<StochVector&>(*svars->x);
	OoqpVector* local_X = (vars_X.vec);
	std::vector<double> local_var(locNx,0.0);
	local_X->copyIntoArray(&local_var[0]);
	//update A , B , C, D matrix
	if(parent == NULL){
		//only B D matrix
		assert(nodeId() == 0);
		MESSAGE("JacFull -- parent is NULL");
		int e_nz = Bmat->numberOfNonZeros();
		int i_nz = Dmat->numberOfNonZeros();
//		MESSAGE("Bmat nz "<<e_nz<<" Dmat nz "<<i_nz);

		CallBackData cbd = {stochInput->prob->userdata,nodeId(),nodeId(),2};

		std::vector<int> e_rowidx(e_nz);
		std::vector<int> e_colptr(locNx+1,0);
		std::vector<double> e_elts(e_nz);

		std::vector<int> i_rowidx(i_nz);
		std::vector<int> i_colptr(locNx+1,0);
		std::vector<double> i_elts(i_nz);
		cbd.typeflag = 2;

#ifdef NLPTIMING
		double stime =MPI_Wtime();
#endif
		stochInput->prob->eval_jac_g(&local_var[0],&local_var[0],
					&e_nz,&e_elts[0],&e_rowidx[0],&e_colptr[0],
					&i_nz,&i_elts[0],&i_rowidx[0],&i_colptr[0],&cbd);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime() - stime;
		gprof.n_jac_g += 1;
#endif

		PRINT_ARRAY("local_var",&local_var[0],locNx);
		PRINT_ARRAY("e_rowidx",e_rowidx,e_nz);
		PRINT_ARRAY("e_colptr",e_colptr,locNx+1);
		PRINT_ARRAY("e_elts",e_elts,e_nz);
		PRINT_ARRAY("i_rowidx",i_rowidx,i_nz);
		PRINT_ARRAY("i_colptr",i_colptr,locNx+1);
		PRINT_ARRAY("i_elts",i_elts,i_nz);

		std::vector<double> e_csr_ret(e_nz,0.0);
		std::vector<double> i_csr_ret(i_nz,0.0);
		convert_to_csr(mB,nB,&e_rowidx[0],&e_colptr[0],&e_elts[0],e_nz,&e_csr_ret[0]);
		convert_to_csr(mD,nD,&i_rowidx[0],&i_colptr[0],&i_elts[0],i_nz,&i_csr_ret[0]);

		Bmat->copyMtxFromDouble(Bmat->numberOfNonZeros(),&e_csr_ret[0]);
		Dmat->copyMtxFromDouble(Dmat->numberOfNonZeros(),&i_csr_ret[0]);
	}
	else{
		//all A B C D
		MESSAGE("JacFull -- with parent");
		std::vector<double> parent_var(parent->locNx,0.0);
		OoqpVector* parent_X = (vars_X.parent->vec);
		parent_X->copyIntoArray(&parent_var[0]);

		int e_nz_Amat = Amat->numberOfNonZeros();
		int i_nz_Cmat = Cmat->numberOfNonZeros();

		CallBackData cbd_link = {stochInput->prob->userdata,nodeId(),parent->stochNode->id(),0};

		std::vector<int> e_amat_rowidx(e_nz_Amat,0.0);
		std::vector<int> e_amat_colptr(parent->locNx+1,0.0);
		std::vector<double> e_amat_elts(e_nz_Amat,0.0);

		std::vector<int> i_cmat_rowidx(i_nz_Cmat,0.0);
		std::vector<int> i_cmat_colptr(parent->locNx+1,0.0);
		std::vector<double> i_cmat_elts(i_nz_Cmat,0.0);

#ifdef NLPTIMING
		double stime = MPI_Wtime();
#endif
		stochInput->prob->eval_jac_g(&parent_var[0],&local_var[0],
				&e_nz_Amat,&e_amat_elts[0],&e_amat_rowidx[0],&e_amat_colptr[0],
				&i_nz_Cmat,&i_cmat_elts[0],&i_cmat_rowidx[0],&i_cmat_colptr[0], &cbd_link);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime() - stime;
		gprof.n_jac_g += 1;
#endif
		PRINT_ARRAY("parent_var",&parent_var[0],parent->locNx);
		PRINT_ARRAY("local_var",&local_var[0],locNx);
		PRINT_ARRAY("e_amat_rowidx",e_amat_rowidx,e_nz_Amat);
		PRINT_ARRAY("e_amat_colptr",e_amat_colptr,parent->locNx+1);
		PRINT_ARRAY("e_amat_elts",e_amat_elts,e_nz_Amat);
		PRINT_ARRAY("i_cmat_rowidx",i_cmat_rowidx,i_nz_Cmat);
		PRINT_ARRAY("i_cmat_colptr",i_cmat_colptr,parent->locNx+1);
		PRINT_ARRAY("i_cmat_elts",i_cmat_elts,i_nz_Cmat);

		int e_nz_Bmat = Bmat->numberOfNonZeros();
		int i_nz_Dmat = Dmat->numberOfNonZeros();

		CallBackData cbd_diag = {stochInput->prob->userdata,nodeId(),nodeId(),2};

		std::vector<int> e_bmat_rowidx(e_nz_Bmat,0.0);
		std::vector<int> e_bmat_colptr(locNx+1,0.0);
		std::vector<double> e_bmat_elts(e_nz_Bmat,0.0);

		std::vector<int> i_dmat_rowidx(i_nz_Dmat,0.0);
		std::vector<int> i_dmat_colptr(locNx+1,0.0);
		std::vector<double> i_dmat_elts(i_nz_Dmat,0.0);

#ifdef NLPTIMING
		stime = MPI_Wtime();
#endif
		stochInput->prob->eval_jac_g(&parent_var[0],&local_var[0],
				&e_nz_Bmat,&e_bmat_elts[0],&e_bmat_rowidx[0],&e_bmat_colptr[0],
				&i_nz_Dmat,&i_dmat_elts[0],&i_dmat_rowidx[0],&i_dmat_colptr[0], &cbd_diag);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime() - stime;
		gprof.n_jac_g += 1;
#endif
		PRINT_ARRAY("e_bmat_rowidx",e_bmat_rowidx,e_nz_Bmat);
		PRINT_ARRAY("e_bmat_colptr",e_bmat_colptr,locNx+1);
		PRINT_ARRAY("e_bmat_elts",e_bmat_elts,e_nz_Bmat);
		PRINT_ARRAY("i_dmat_rowidx",i_dmat_rowidx,i_nz_Dmat);
		PRINT_ARRAY("i_dmat_colptr",i_dmat_colptr,locNx+1);
		PRINT_ARRAY("i_dmat_elts",i_dmat_elts,i_nz_Dmat);

		std::vector<double> e_amat_csr(e_nz_Amat,0.0);
		std::vector<double> i_cmat_csr(i_nz_Cmat,0.0);
		std::vector<double> e_bmat_csr(e_nz_Bmat,0.0);
		std::vector<double> i_dmat_csr(i_nz_Dmat,0.0);
		convert_to_csr(mA,nA,&e_amat_rowidx[0],&e_amat_colptr[0],&e_amat_elts[0],e_nz_Amat,&e_amat_csr[0]);
		convert_to_csr(mC,nC,&i_cmat_rowidx[0],&i_cmat_colptr[0],&i_cmat_elts[0],i_nz_Cmat,&i_cmat_csr[0]);
		convert_to_csr(mB,nB,&e_bmat_rowidx[0],&e_bmat_colptr[0],&e_bmat_elts[0],e_nz_Bmat,&e_bmat_csr[0]);
		convert_to_csr(mD,nD,&i_dmat_rowidx[0],&i_dmat_colptr[0],&i_dmat_elts[0],i_nz_Dmat,&i_dmat_csr[0]);

		Amat->copyMtxFromDouble(Amat->numberOfNonZeros(),&e_amat_csr[0]);
		Bmat->copyMtxFromDouble(Bmat->numberOfNonZeros(),&e_bmat_csr[0]);
		Cmat->copyMtxFromDouble(Cmat->numberOfNonZeros(),&i_cmat_csr[0]);
		Dmat->copyMtxFromDouble(Dmat->numberOfNonZeros(),&i_dmat_csr[0]);
	}

	for(size_t it=0; it<children.size(); it++)
		children[it]->JacFull(svars->children[it], NULL,NULL);

	MESSAGE("exit JacFull");

}


void StructJuMPsInfo::Hessian(NlpGenVars * nlpvars, SymMatrix *Hess)
{
	MESSAGE("enter Hessian");
	//update Qdiag and Qborder
	long long mqi, nqi, mqb, nqb;
	Qdiag->getSize(mqi,nqi);
	Qborder->getSize(mqb,nqb);
//	MESSAGE(" Qdiag "<<mqi<<"  "<<nqi<<"  nz"<<Qdiag->numberOfNonZeros());
//	MESSAGE(" Qborder "<<mqb<<"  "<<nqb<<"  nz"<<Qborder->numberOfNonZeros());

	sVars * vars = dynamic_cast<sVars*>(nlpvars);
	StochVector& vars_X = dynamic_cast<StochVector&>(*vars->x);
	StochVector& vars_Y = dynamic_cast<StochVector&>(*vars->y);
	StochVector& vars_Z = dynamic_cast<StochVector&>(*vars->z);
	OoqpVector* local_X = vars_X.vec;
	OoqpVector* local_Y = vars_Y.vec; //eq con
	OoqpVector* local_Z = vars_Z.vec; //ieq con

	std::vector<double> local_var(locNx,0.0);
	local_X->copyIntoArray(&local_var[0]);
	std::vector<double> local_y(locMy,0.0);
	std::vector<double> local_z(locMz,0.0);
	local_Y->copyIntoArray(&local_y[0]);
	local_Z->copyIntoArray(&local_z[0]);
	std::vector<double> lam(locMy+locMz,0.0);
	int i=0;
	for(i=0;i<locMy;i++) lam[i] = -local_y[i];
	for(;i<locMy+locMz;i++) lam[i] = -local_z[i-locMy];

	int nzqd = Qdiag->numberOfNonZeros();
	std::vector<double> elts(nzqd,0.0);

	if(gmyid == 0) {
		MESSAGE("gmyid="<<gmyid);
		std::vector<int> rowidx(nzqd,0.0);
		std::vector<int> colptr(locNx+1,0.0);
		CallBackData cbd = {stochInput->prob->userdata,0,0,2};
#ifdef NLPTIMING
		double stime = MPI_Wtime();
#endif
		stochInput->prob->eval_h(&local_var[0],&local_var[0],&lam[0],&nzqd,&elts[0],&rowidx[0],&colptr[0],&cbd);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime() - stime;
		gprof.n_laghess += 1;
#endif
		PRINT_ARRAY("local_var",local_var,locNx);
		PRINT_ARRAY("lam",lam,locMy+locMz);
		PRINT_ARRAY("rowidx",rowidx,nzqd);
		PRINT_ARRAY("colptr",colptr,locNx+1);
		PRINT_ARRAY("elts",elts,nzqd);
	}

	for(size_t it=0; it<children.size(); it++){
		MESSAGE("it - "<<it);
		children[it]->Hessian_FromSon(vars->children[it],&elts[0]);
	}
	PRINT_ARRAY("elts",elts,nzqd);

	//MPI ALL REDUCE
	std::vector<double> g_elts(nzqd,0.0);
	MPI_Allreduce(&elts[0], &g_elts[0], nzqd, MPI_DOUBLE, MPI_SUM, mpiComm);
	PRINT_ARRAY("after reudce - g_elts",g_elts,nzqd);

	Qdiag->copyMtxFromDouble(nzqd,&g_elts[0]);
	MESSAGE("exit Hessian");
}

void StructJuMPsInfo::Hessian_FromSon(NlpGenVars* nlpvars, double *parent_hess){
	MESSAGE("enter Hessian_FromSon - "<<nodeId());
	long long mqi, nqi, mqb, nqb;
	Qdiag->getSize(mqi,nqi);
	Qborder->getSize(mqb,nqb);
//	MESSAGE(" Qdiag "<<mqi<<"  "<<nqi<<"  nz"<<Qdiag->numberOfNonZeros());
//	MESSAGE(" Qborder "<<mqb<<"  "<<nqb<<"  nz"<<Qborder->numberOfNonZeros());

	sVars * vars = dynamic_cast<sVars*>(nlpvars);
	StochVector& vars_X = dynamic_cast<StochVector&>(*vars->x);
	StochVector& vars_Y = dynamic_cast<StochVector&>(*vars->y);
	StochVector& vars_Z = dynamic_cast<StochVector&>(*vars->z);
	OoqpVector* local_X = vars_X.vec;
	OoqpVector* local_Y = vars_Y.vec; //eq con
	OoqpVector* local_Z = vars_Z.vec; //ieq con

	std::vector<double> local_var(locNx,0.0);
	local_X->copyIntoArray(&local_var[0]);
	std::vector<double> local_y(locMy,0.0);
	std::vector<double> local_z(locMz,0.0);
	local_Y->copyIntoArray(&local_y[0]);
	local_Z->copyIntoArray(&local_z[0]);
	std::vector<double> lam(locMy+locMz,0.0);
	int i=0;
	for(i=0;i<locMy;i++) lam[i] = -local_y[i];
	for(;i<locMy+locMz;i++) lam[i] = -local_z[i-locMy];

	std::vector<double> parent_var(parent->locNx,0.0);
	OoqpVector* parent_X = (vars_X.parent->vec);
	parent_X->copyIntoArray(&parent_var[0]);

	PRINT_ARRAY("parent_var",parent_var,parent->locNx);
	PRINT_ARRAY("local_var",local_var,locNx);
	PRINT_ARRAY("lam",lam,locMy+locMz);

  //nzqd
  {
    MESSAGE(" --- Child diagonal");
    int nzqd = Qdiag->numberOfNonZeros();
    std::vector<double> elts(nzqd,0.0);
    std::vector<int> rowidx(nzqd,0.0);
    std::vector<int> colptr(locNx+1,0.0);
    CallBackData cbd_nzqd = {stochInput->prob->userdata,nodeId(),nodeId(),0};
#ifdef NLPTIMING
    double stime = MPI_Wtime();
#endif
    stochInput->prob->eval_h(&parent_var[0],&local_var[0],&lam[0],&nzqd,&elts[0],&rowidx[0],&colptr[0],&cbd_nzqd);
#ifdef NLPTIMING
    gprof.t_model_evaluation += MPI_Wtime() - stime;
    gprof.n_laghess += 1;
#endif
    PRINT_ARRAY("rowidx",rowidx,nzqd);
    PRINT_ARRAY("colptr",colptr,locNx+1);
    PRINT_ARRAY("elts",elts,nzqd);
    Qdiag->copyMtxFromDouble(nzqd,&elts[0]);
  }

	//nzqb
	{
		MESSAGE(" --- linking border");
		int nzqb = Qborder->numberOfNonZeros();
		std::vector<double> elts(nzqb,0.0);
		std::vector<int> rowidx(nzqb,0.0);
		std::vector<int> colptr(parent->locNx+1,0.0);
		CallBackData cbd_nzqb = {stochInput->prob->userdata,0,nodeId(),0};
#ifdef NLPTIMING
		double stime = MPI_Wtime();
#endif
		stochInput->prob->eval_h(&parent_var[0],&local_var[0],&lam[0],&nzqb,&elts[0],&rowidx[0],&colptr[0],&cbd_nzqb);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime() - stime;
		gprof.n_laghess += 1;
#endif
		PRINT_ARRAY("rowidx",rowidx,nzqb);
		PRINT_ARRAY("colptr",colptr,parent->locNx+1);
		PRINT_ARRAY("elts",elts,nzqb);

		std::vector<double>	csr_ret(nzqb,0.0);
		convert_to_csr(locNx,parent->locNx,&rowidx[0],&colptr[0],&elts[0],nzqb,&csr_ret[0]);
		Qborder->copyMtxFromDouble(nzqb,&csr_ret[0]);
	}

	//pnzqd
  {
    MESSAGE("  -- Parent contribution - ");
    int pnzqd = parent->Qdiag->numberOfNonZeros();
    std::vector<double> elts(pnzqd,0.0);
    std::vector<int> rowidx(pnzqd,0.0);
    std::vector<int> colptr(parent->locNx+1,0.0);
    CallBackData cbd_pnzqd = {stochInput->prob->userdata,nodeId(),0,2};
#ifdef NLPTIMING
    double stime = MPI_Wtime();
#endif
    stochInput->prob->eval_h(&parent_var[0],&local_var[0],&lam[0],&pnzqd,&elts[0],&rowidx[0],&colptr[0],&cbd_pnzqd);
#ifdef NLPTIMING
    gprof.t_model_evaluation += MPI_Wtime() - stime;
    gprof.n_laghess += 1;
#endif
    PRINT_ARRAY("rowidx",rowidx,pnzqd);
    PRINT_ARRAY("colptr",colptr,parent->locNx+1);
    PRINT_ARRAY("elts",elts,pnzqd);
    for(int i=0;i<pnzqd;i++) parent_hess[i] += elts[i];
  }

	MESSAGE("exit Hessian_FromSon");
}

void StructJuMPsInfo::get_InitX0(OoqpVector* vX){
	MESSAGE("enter get_InitX0 - "<<nodeId());
	StochVector* vars_X = dynamic_cast<StochVector*>(vX);
	OoqpVector* local_X = vars_X->vec;
//	assert(locNx == vX->n);
	assert(children.size() == vars_X->children.size());

	std::vector<double> temp_var(locNx,0.0);
	CallBackData cbd = {stochInput->prob->userdata,nodeId(),nodeId(),0};
#ifdef NLPTIMING
	double stime = MPI_Wtime();
#endif
	stochInput->prob->init_x0(&temp_var[0],&cbd);
#ifdef NLPTIMING
	gprof.t_model_evaluation += MPI_Wtime() - stime;
	gprof.n_init_x0 += 1;
#endif
	PRINT_ARRAY("temp_var",temp_var,locNx);
	IF_VERBOSE_DO( local_X->print(); );
	local_X->copyFromArray(&temp_var[0]);
	IF_VERBOSE_DO( local_X->print(); );
	for(size_t it=0; it<children.size(); it++)
	    children[it]->get_InitX0(vars_X->children[it]);

	MESSAGE("exit get_InitX0");
}

void StructJuMPsInfo::writeSolution(NlpGenVars* nlpvars)
{
	MESSAGE("writeSolution");
	sVars * vars = dynamic_cast<sVars*>(nlpvars);
	StochVector& vars_X = dynamic_cast<StochVector&>(*vars->x);
	StochVector& vars_Y = dynamic_cast<StochVector&>(*vars->y);
	StochVector& vars_Z = dynamic_cast<StochVector&>(*vars->z);
	OoqpVector* local_X = vars_X.vec;
	OoqpVector* local_Y = vars_Y.vec; //eq con
	OoqpVector* local_Z = vars_Z.vec; //ieq con

	std::vector<double> local_var(locNx,0.0);
	local_X->copyIntoArray(&local_var[0]);
	std::vector<double> local_y(locMy,0.0);
	std::vector<double> local_z(locMz,0.0);
	local_Y->copyIntoArray(&local_y[0]);
	local_Z->copyIntoArray(&local_z[0]);

	CallBackData cbd = {stochInput->prob->userdata,nodeId(),nodeId(),0};
#ifdef NLPTIMING
		double stime = MPI_Wtime();
#endif
	stochInput->prob->write_solution(&local_var[0],&local_y[0],&local_z[0], &cbd);
#ifdef NLPTIMING
		gprof.t_model_evaluation += MPI_Wtime() - stime;
		gprof.n_write_solution += 1;
#endif

	for(size_t it=0; it<children.size(); it++){
		MESSAGE("it - "<<it);
		children[it]->writeSolution(vars->children[it]);
	}
	MESSAGE("end writeSolution");
}
