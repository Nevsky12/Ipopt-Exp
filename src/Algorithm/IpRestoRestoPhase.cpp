// Copyright (C) 2004, 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2005-02-11

#include "IpRestoRestoPhase.hpp"
#include "IpRestoIpoptNLP.hpp"
#include "IpRestoIterateInitializer.hpp"

namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

RestoRestorationPhase::RestoRestorationPhase()
{ }

RestoRestorationPhase::~RestoRestorationPhase()
{ }

bool RestoRestorationPhase::InitializeImpl(
   const OptionsList& /*options*/,
   const std::string& /*prefix*/
)
{
   return true;
}

bool RestoRestorationPhase::PerformRestoration()
{
   DBG_START_METH("RestoRestorationPhase::PerformRestoration",
                  dbg_verbosity);
   Jnlst().Printf(J_DETAILED, J_MAIN,
                  "Performing second level restoration phase for current constriant violation %8.2e\n",
                  IpCq().curr_constraint_violation());

   DBG_ASSERT(IpCq().curr_constraint_violation() > 0.);

   // Get a grip on the restoration phase NLP and obtain the pointers
   // to the original NLP data
   SmartPtr<RestoIpoptNLP> resto_ip_nlp = static_cast<RestoIpoptNLP*>(&IpNLP());
   DBG_ASSERT(dynamic_cast<RestoIpoptNLP*> (&IpNLP()));
   SmartPtr<IpoptNLP> orig_ip_nlp = static_cast<IpoptNLP*>(&resto_ip_nlp->OrigIpNLP());
   DBG_ASSERT(dynamic_cast<IpoptNLP*> (&resto_ip_nlp->OrigIpNLP()));

   // Get the current point and create a new vector for the result
   SmartPtr<const CompoundVector> Ccurr_x = static_cast<const CompoundVector*>(GetRawPtr(IpData().curr()->x()));
   DBG_ASSERT(dynamic_cast<const CompoundVector*> (GetRawPtr(IpData().curr()->x())));
   SmartPtr<const CompoundVector> Ccurr_s = static_cast<const CompoundVector*>(GetRawPtr(IpData().curr()->s()));
   DBG_ASSERT(dynamic_cast<const CompoundVector*> (GetRawPtr(IpData().curr()->s())));
   DBG_ASSERT(Ccurr_s->NComps() == 1);
   SmartPtr<Vector> new_x = IpData().curr()->x()->MakeNew();
   SmartPtr<CompoundVector> Cnew_x = static_cast<CompoundVector*>(GetRawPtr(new_x));

   // The x values remain unchanged
   SmartPtr<Vector> x = Cnew_x->GetCompNonConst(0);
   x->Copy(*Ccurr_x->GetComp(0));

   // ToDo in free mu mode - what to do here?
   Number mu = IpData().curr_mu();

   // Compute the initial values for the n and p variables for the
   // equality constraints
   Number rho = resto_ip_nlp->Rho();
   SmartPtr<Vector> nc = Cnew_x->GetCompNonConst(1);
   SmartPtr<Vector> pc = Cnew_x->GetCompNonConst(2);
   SmartPtr<const Vector> cvec = orig_ip_nlp->c(*Ccurr_x->GetComp(0));
   RestoIterateInitializer::ComputeSlackVariables(
      mu / (2. * rho), *cvec, *nc, *pc);
   DBG_PRINT_VECTOR(2, "nc", *nc);
   DBG_PRINT_VECTOR(2, "pc", *pc);

   // initial values for the n and p variables for the inequality
   // constraints
   SmartPtr<Vector> nd = Cnew_x->GetCompNonConst(3);
   SmartPtr<Vector> pd = Cnew_x->GetCompNonConst(4);
   SmartPtr<Vector> dvec = pd->MakeNew();
   dvec->Copy(*orig_ip_nlp->d(*Ccurr_x->GetComp(0)));
   dvec->Axpy(-1., *Ccurr_s->GetComp(0));
   RestoIterateInitializer::ComputeSlackVariables(
      mu / (2. * rho), *dvec, *nd, *pd);
   DBG_PRINT_VECTOR(2, "nd", *nd);
   DBG_PRINT_VECTOR(2, "pd", *pd);

   // Now set the trial point to the solution of the restoration phase
   // s and all multipliers remain unchanged
   SmartPtr<IteratesVector> new_trial = IpData().curr()->MakeNewContainer();
   new_trial->Set_x(*new_x);
   IpData().set_trial(new_trial);

   IpData().Append_info_string("R");

   return true;
}

} // namespace Ipopt
