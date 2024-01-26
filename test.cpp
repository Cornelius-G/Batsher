#include "SHERPA/Main/Sherpa.H"
#include "SHERPA/Initialization/Initialization_Handler.H"
#include "ATOOLS/Org/CXXFLAGS.H"
#include "ATOOLS/Org/CXXFLAGS_PACKAGES.H"
#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Org/Message.H"
#include "ATOOLS/Org/My_MPI.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Math/Random.H"
#include "AddOns/Python/MEProcess.H"
#include "HepMC3/GenEvent.h"
#include "HepMC3/WriterAscii.h"
#include "HepMC3/Print.h"
#include "HepMC3/Attribute.h"
#include "Rambo.hpp"

double eval_me(MEProcess& process, ATOOLS::Vec4D_Vector& momenta)
{
    process.SetMomenta(momenta);
    return process.CSMatrixElement();
}

void correct_momenta(ATOOLS::Vec4D_Vector& p) {
  int nin = 2;
  int nvec = p.size();
  ATOOLS::Vec4D  momsum(0.,0.,0.,0.);
  size_t imax(0);
  double Emax(0.0);
  for (size_t i(0); i < nin; ++i)
    momsum += -p[i];
  for (size_t i(nin); i < nvec; ++i) {
    momsum += p[i];
    if (p[i][0] > Emax) {
      Emax = p[i][0];
      imax = i;
    }
    p[i][0] = sqrt(p[i].PSpat2());
  }
  p[imax] -= momsum;
  p[imax][0] = sqrt(p[imax].PSpat2());
  double E0tot(0);
  for (size_t i(0); i < nvec; ++i)
    E0tot += (i < nin ? -1. : 1.) * p[i][0];
  double p2[2] = {p[0].PSpat2(), p[1].PSpat2()};
  double E0[2] = {-p[0][0], -p[1][0]};
  double E1[2] = {p2[0] / E0[0],
                  -(ATOOLS::Vec3D(p[0]) * ATOOLS::Vec3D(p[1])) / E0[1]};
  double E1tot = E1[0] + E1[1];
  double E2tot = (p2[0] - (E1[1])*(E1[1])) / (2 * E0[1]);
  double eps1 = -E0tot / E1tot;
  double eps = eps1 * (1 - eps1 * E2tot / E1tot);
  p[1] = -p[1] + p[0] * eps;
  p[0] = -p[0] - p[0] * eps;
  for (int i(0); i < 2; ++i) {
    p[i][0] = -std::abs(p[i][3]);
  }
  for (size_t i(0); i < nin; ++i)
    p[i] = -p[i];
}

void conserve_momentum(ATOOLS::Vec4D_Vector& p, double ET)
{
  int itmax = 6;
  double accu  = ET * 1.e-14; //pow(10.,-14.);

  double x = 1.;

  int nin = 2;
  int nout = p.size() - nin;

  // Loop to calculate x

  double f0,g0,x2;
  short int iter = 0; 
  std::vector<double> E(p.size());
  for (;;) {
    f0 = -ET;g0 = 0.;x2 = x*x;
    for (short int i=nin;i<nin+nout;i++) {
      E[i] = sqrt(x2*p[i][0]*p[i][0]);
      f0  += E[i];
      g0  += p[i][0]*p[i][0]/E[i];
    }
    if (abs(f0)<accu) break; 
    iter++;
    if (iter>itmax) break;
    x -= f0/(x*g0);  
  }
  
  // Construct Momenta
  for (short int i=nin;i<nin+nout;i++) p[i] = ATOOLS::Vec4D(E[i],x*ATOOLS::Vec3D(p[i]));
}

//----- Main ------------------------------------------------------------------
int main(int argc,char* argv[])
{

// Get flag whether program is called from Julia or not
bool juliaFlag = false;
int shiftCount = 0;
// Start from 1 because argv[0] is the program name
for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "julia") {
        shiftCount++;
        juliaFlag = true;
        //break; // Optional: Stop searching after "julia" is found
    }
    argv[i - shiftCount] = argv[i];
}
argc -= shiftCount; // This is so the program behaves as if julia never existed for the rest of the flags

#ifdef USING__MPI
  MPI_Init(&argc, &argv);
#endif
  // initialize the framework
  try {
    SHERPA::Sherpa *Generator(new SHERPA::Sherpa(argc, argv));
    Generator->InitializeTheRun();

    // create a MEProcess instance
    MEProcess Process(Generator);
    Process.Initialize();

    ATOOLS::ran->SetSeed(42);
    ATOOLS::rpa->gen.SetNumberOfEvents(50000);

    // HepMC3 Interface
    std::string hepmc_path;
    if (juliaFlag) {
      hepmc_path = "output/events_bat.hepmc"; // File path if flag is true
    } else {
      hepmc_path = "output/events_sherpa.hepmc"; // File path if flag is false
    }

    HepMC3::WriterAscii output_file(hepmc_path.c_str());

    int nparticles = Process.GetProc()->NOut();
    int ndim = 3*nparticles-4;
    double Ecm = ATOOLS::rpa->gen.Ecms();
    long int nevents = ATOOLS::rpa->gen.NumberOfEvents();
    
    //msg_out()<<"nevents: "<<ATOOLS::rpa->gen.NumberOfEvents()<<std::endl;
    //msg_out()<<"Ecm: "<<ATOOLS::rpa->gen.Ecms()<<std::endl;

    ATOOLS::Vec4D pin0(Ecm/2., 0., 0., Ecm/2.);
    ATOOLS::Vec4D pin1(Ecm/2., 0., 0., -Ecm/2.);
    ATOOLS::Vec4D_Vector pout;

    for (unsigned int i=0; i<nparticles; i++) {
        pout.push_back(ATOOLS::Vec4D());
    }

    // instantiate Rambo mapping
    Rambo rambo(nparticles, Ecm);

    std::vector<double> rans(ndim);
    ATOOLS::Vec4D_Vector momenta(nparticles+2);
    momenta[0] = pin0;
    momenta[1] = pin1;
    double cs_me(0.);
    double ps = rambo.weight();
    double w(0.);
    double sumw(0.);
    double sumw2(0.);
    int ntrials_total(0);
    int ntrials_this(0);
    int triggers(0);
    double xs(0.), err(0.);
    std::vector<double> xs_vec(4), err_vec(4);
    bool accept;

    // construct HepMC event
    HepMC3::GenEvent evt(HepMC3::Units::GEV,HepMC3::Units::MM);
    evt.set_run_info(std::make_shared<HepMC3::GenRunInfo>());
    //                                                                                       px       py       pz       e            pdgid        status
    HepMC3::GenParticlePtr p1 = std::make_shared<HepMC3::GenParticle>( HepMC3::FourVector( pin0[1], pin0[2], pin0[3], pin0[0] ), Process.GetFlav(0), 3 );
    HepMC3::GenParticlePtr p2 = std::make_shared<HepMC3::GenParticle>( HepMC3::FourVector( pin1[1], pin1[2], pin1[3], pin1[0] ), Process.GetFlav(1), 3 );
    std::vector<HepMC3::GenParticlePtr> hepmc_out_particles;
    for (size_t i(2); i<nparticles+2; ++i) {
      hepmc_out_particles.push_back(std::make_shared<HepMC3::GenParticle>( HepMC3::FourVector( 0.0, 0.0, 0.0, 0.0 ), Process.GetFlav(i), 3 ));
    }

    HepMC3::GenVertexPtr v1 = std::make_shared<HepMC3::GenVertex>();
    v1->add_particle_in(p1);
    v1->add_particle_in(p2);
    for (auto particle : hepmc_out_particles) {
      v1->add_particle_out(particle);	    
    }
    evt.add_vertex(v1);

    std::shared_ptr<HepMC3::GenCrossSection> cross_section = std::make_shared<HepMC3::GenCrossSection>();
    evt.add_attribute("GenCrossSection", cross_section);

    std::shared_ptr<HepMC3::VectorDoubleAttribute> rambo_input = std::make_shared<HepMC3::VectorDoubleAttribute>();
    evt.add_attribute("RamboInput", rambo_input);

    std::vector<std::string> w_names = {"Weight", "EXTRA__MEWeight", "EXTRA__PSWeight", "EXTRA__NTrials"};
    evt.run_info()->set_weight_names(w_names);
    evt.weights() = std::vector<double>(4);

    // while loop to keep asking for testpoints
    bool stopRequested = false;
    // File ----------------------------------------------
    std::ofstream outputFile("output/example.txt");

    // Check if the file was successfully opened
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Write data to the file

    // ---------------------------------------------------

    //for (size_t i(0); i<200000; ++i) {
    //for (size_t i(0); i<nevents; ++i) {
    int i = 0;
    while (!stopRequested && (i < nevents || juliaFlag)) {
      // Prompt the user for testpoint values
      std::vector<double> testpoint;
      std::string inputVal;

      //std::cout << "Enter testpoint values (e.g., 0.15 0.15 ....): ";
      while (juliaFlag && std::cin >> inputVal) {
          if (inputVal == "s") {
              stopRequested = true;
              break;
          }
          double number = std::stod(inputVal);
          outputFile << "inputVal: "<< number << "  " << std::flush;
          testpoint.push_back(number);
          if (testpoint.size() >= 5) {
              break; // Break after 5 values are entered
          }
      }
      if(stopRequested){continue;}
      
      ntrials_total++;
      ntrials_this++;
      for (auto& r : rans) {
        r = ATOOLS::ran->Get();
      }

      // map from hypercube to momenta
      if(juliaFlag){rambo.map(testpoint, pout);}else{rambo.map(rans, pout);}

      std::copy(pout.begin(), pout.end(), momenta.begin()+2);

      correct_momenta(momenta);
      conserve_momentum(momenta, Ecm);

      Process.SetMomenta(momenta);

      // count triggers
      accept = Process.GetProc()->Trigger(momenta);

      triggers += accept;

      if (accept) {
        if(juliaFlag){rambo_input->set_value(testpoint);}else{rambo_input->set_value(rans);}
        
        // get matrix elements
        cs_me = Process.CSMatrixElement();
        
        // print some information
        if(juliaFlag){
        outputFile << "  cs_me: "<< cs_me << std::endl;
        std::cout << std::setprecision(16) << cs_me << std::endl;}
        else{
          if(i%10000 == 0)std::cout << i << " Events generated" << std::endl;
        }

        w = cs_me*ps;
        sumw += w;
        sumw2 += w*w;

        for (size_t i(0); i<nparticles; ++i) {
        hepmc_out_particles[i]->set_momentum(HepMC3::FourVector( pout[i][1], pout[i][2], pout[i][3], pout[i][0] ));	
        }

        evt.weights()[0] = Process.GetFlux() * w * ATOOLS::rpa->Picobarn();
        evt.weights()[1] = cs_me;
        evt.weights()[2] = ps;
        evt.weights()[3] = ntrials_this;

        // calculate cross-section
        xs = Process.GetFlux() * sumw/ntrials_total * ATOOLS::rpa->Picobarn();
        err = Process.GetFlux() * sqrt(abs(sumw2/ntrials_total - (sumw/ntrials_total)*(sumw/ntrials_total))/(ntrials_total-1.0)) * ATOOLS::rpa->Picobarn();
        //cross_section->set_cross_section(xs, err);
        xs_vec[0] = xs;
        err_vec[0] = err;
        cross_section->set_cross_section(xs_vec, err_vec);
    
        // write to file
        output_file.write_event(evt);

        evt.set_event_number(evt.event_number()+1);
        ntrials_this = 0;   
        i++;
      } else {
        if(juliaFlag){std::cout << 0 << std::endl;} //Return Matrixelement for non triggered events
      }
  
    }

    // info strings
    //size_t precision(msg_out().precision());
    //msg_out().precision(16);
    //msg_out()<<"cross-section:                   "<<Process.GetFlux() * sumw/nevents * ATOOLS::rpa->Picobarn()<<std::endl;
    //msg_out()<<"ntrials_total:                   "<<ntrials_total<<std::endl;
    //msg_out()<<"triggers:                        "<<triggers<<std::endl;
    //msg_out().precision(precision);
    // Close the file
    outputFile.close();
    delete Generator;
  }
  catch (const ATOOLS::normal_exit& exception) {
    msg_Error() << exception << std::endl;
    ATOOLS::exh->Terminate(0);
  }
  catch (const ATOOLS::Exception& exception) {
    msg_Error() << exception << std::endl;
    ATOOLS::exh->Terminate(1);
  }
  catch (const std::exception& exception) {
    msg_Error() << exception.what() << std::endl;
    ATOOLS::exh->Terminate(1);
  }

#ifdef USING__MPI
  MPI_Finalize();
#endif

  return 0;
}
