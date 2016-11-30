#!/bin/bash
#
#warning: note that this script does not print the 'exact' usage of the target accounts

jiadeng_fluxoe_users=$(mdiag -a jiadeng_fluxoe | grep Users);
jiadeng_fluxoe_users=${jiadeng_fluxoe_users:9};

jiadeng_fluxg_users=$(mdiag -a jiadeng_fluxg | grep Users);
jiadeng_fluxg_users=${jiadeng_fluxg_users:9};

jiadeng_flux_users=$(mdiag -a jiadeng_flux | grep Users);
jiadeng_flux_users=${jiadeng_flux_users:9};

# echo "jiadeng_fluxoe usage";
echo "fluxoe usage";
printf "\n";
qstat -a fluxoe | grep ${jiadeng_fluxoe_users//,/\\|};
printf "\n";

# echo "jiadeng_fluxg usage";
echo "fluxg usage";
printf "\n";
qstat -a fluxg | grep ${jiadeng_fluxg_users//,/\\|};
printf "\n";

# echo "jiadeng_flux usage";
echo "flux usage";
printf "\n";
qstat -a flux | grep ${jiadeng_flux_users//,/\\|};
printf "\n";
