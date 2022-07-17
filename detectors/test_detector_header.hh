#ifndef HkIWs
#define HkIWs 1

#include "VDetector.hh"

using namespace std;

class HkIWs : public VDetector {
    public:
        HkIWs() {
            cout << "You are currently using the detector HkIWs."
            << endl;

            Initialization();
        };
        virtual ~HkIWs(){};

        virtual void Initialization() {
            g1                   = 0.117;
            sPEres               = 0.37;
            sPEthr               = 0.3845901639344262;
            sPEeff               = 1.0;
            noiseBaseline[0]     = 0.0;
            noiseBaseline[1]     = 0.08;
            noiseBaseline[2]     = 0.0;
            noiseBaseline[3]     = 0.0;
            P_dphe               = 0.173;
            coinWind             = 100;
            coinLevel            = 2;
            numPMTs              = 119;
            OldW13eV             = true;
            noiseLinear[0]       = 0.0;
            noiseLinear[1]       = 0.0;
            g1_gas               = 0.1;
            s2Fano               = 3.6;
            s2_thr               = 192.29508196721312;
            E_gas                = 6.25;
            eLife_us             = 800.0;
            T_Kelvin             = 173.0;
            p_bar                = 1.57;
            dtCntr               = 160.0;
            dt_min               = 38.0;
            dt_max               = 305.0;
            radius               = 200.0;
            radmax               = 235.0;
            TopDrift             = 544.95;
            anode                = 549.2;
            gate                 = 539.2;
            cathode              = 55.9;
            PosResExp            = 0.015;
            PosResBase           = 70.8364;
        }
