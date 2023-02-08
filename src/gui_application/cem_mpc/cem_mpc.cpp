#include "cem_mpc.h"
#include "cem_mpc_inl.h"
#include "common/dynamics.h"

namespace mpex {

template class CEM_MPC<EigenKinematicBicycle>;
template class CEM_MPC<CurvilinearKinematicBicycle>;

} // namespace mpex