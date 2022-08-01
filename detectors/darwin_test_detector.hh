#ifndef darwin_test_detector_hh
#define darwin_test_detector_hh 1

#include "VDetector.hh"

using namespace std;

class darwin_test_detector : public VDetector {
    public:
        darwin_test_detector() {

            Initialization();
        };
        ~darwin_test_detector() override = default;

        void Initialization() override {
            g1                   =   0.15;
            sPEres               =   0.37;
            sPEthr               =   0.35;
            sPEeff               =   0.9;
            noiseBaseline[0]     =   0.0;
            noiseBaseline[1]     =   0.0;
            noiseBaseline[2]     =   0.0;
            noiseBaseline[3]     =   0.0;
            P_dphe               =   0.2;
            coinWind             =   100;
            coinLevel            =   3;
            numPMTs              =   494;
            OldW13eV             =   true;
            noiseLinear[0]       =   0.0;
            noiseLinear[1]       =   0.0;
            g1_gas               =   0.1;
            s2Fano               =   3.6;
            s2_thr               =   100;
            E_gas                =   10.0;
            eLife_us             =   5000.0;
            T_Kelvin             =   175.0;
            p_bar                =   2.0;
            dtCntr               =   822.0;
            dt_min               =   75.8;
            dt_max               =   1536.5;
            radius               =   1300.0;
            radmax               =   1350.0;
            TopDrift             =   3005.0;
            anode                =   3012.5;
            gate                 =   3000.0;
            cathode              =   250.0;
            PosResExp            =   0.015;
            PosResBase           =   30.0;
        };
    };
#endif
