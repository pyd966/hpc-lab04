#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <string>
#include <cmath>
#include <map>
using namespace std;

#include <mpi.h>

#include "misc.h"
#include "macrodef.h"

#ifdef USE_GPU
#include "bssn_gpu_class.h"
#else
#include "bssn_class.h"

#ifdef __linux__
static int abe_affinity_cpu_count(const cpu_set_t &cpus)
{
      int count = 0;
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &cpus))
                  ++count;
      return count;
}

static bool abe_parse_cpu_list(const char *text, cpu_set_t &cpus)
{
      CPU_ZERO(&cpus);
      if (text == NULL || *text == '\0')
            return false;
      const char *cursor = text;
      bool found_cpu = false;
      while (*cursor != '\0')
      {
            errno = 0;
            char *end = NULL;
            long first = strtol(cursor, &end, 10);
            if (errno != 0 || end == cursor || first < 0 || first >= CPU_SETSIZE)
                  return false;
            long last = first;
            if (*end == '-')
            {
                  cursor = end + 1;
                  errno = 0;
                  last = strtol(cursor, &end, 10);
                  if (errno != 0 || end == cursor || last < first || last >= CPU_SETSIZE)
                        return false;
            }
            for (long cpu = first; cpu <= last; ++cpu)
                  CPU_SET(cpu, &cpus);
            found_cpu = true;
            if (*end == '\0')
                  break;
            if (*end != ',')
                  return false;
            cursor = end + 1;
      }
      return found_cpu;
}

static void abe_restore_and_report_affinity(char *const argv[])
{
      cpu_set_t inherited;
      cpu_set_t active;
      const bool have_inherited = sched_getaffinity(0, sizeof(inherited), &inherited) == 0;
      const char *requested = getenv("AMSS_SCHEDULER_CPU_LIST");
      const char *restore_status = "not-requested";
      int restore_errno = 0;
      if (requested != NULL && *requested != '\0')
      {
            cpu_set_t target;
            if (!abe_parse_cpu_list(requested, target))
                  restore_status = "invalid-target";
            else if (sched_setaffinity(0, sizeof(target), &target) != 0)
            {
                  restore_status = "failed";
                  restore_errno = errno;
            }
            else
                  restore_status = "ok";
      }
      if (strcmp(restore_status, "ok") == 0 &&
          getenv("AMSS_AFFINITY_PREPARED") == NULL)
      {
            if (setenv("AMSS_AFFINITY_PREPARED", "1", 1) == 0)
            {
                  execv("/proc/self/exe", argv);
                  restore_errno = errno;
                  restore_status = "re-exec-failed";
                  unsetenv("AMSS_AFFINITY_PREPARED");
            }
      }
      const bool have_active = sched_getaffinity(0, sizeof(active), &active) == 0;
      cout << "==> ABE affinity: inherited="
           << (have_inherited ? abe_affinity_cpu_count(inherited) : -1)
           << " target=" << ((requested != NULL && *requested != '\0') ? requested : "unset")
           << " active=" << (have_active ? abe_affinity_cpu_count(active) : -1)
           << " restore=" << restore_status;
      if (restore_errno != 0)
            cout << " errno=" << restore_errno << " (" << strerror(restore_errno) << ")";
      cout << endl;
}
#endif
#endif

namespace parameters
{
      map<string, int> int_par;
      map<string, double> dou_par;
      map<string, string> str_par;
}

//=================================================================================================
//=================================================================================================

int main(int argc, char *argv[])
{
      int myrank = 0, nprocs = 1;
#if defined(__linux__) && !defined(USE_GPU)
      abe_restore_and_report_affinity(argv);
#endif
      MPI_Init(&argc, &argv);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#ifdef _OPENMP
      const int num_places = omp_get_num_places();
      int place_cpus = 0;
      for (int place = 0; place < num_places; ++place)
            place_cpus += omp_get_place_num_procs(place);
      if (myrank == 0)
            cout << "==> ABE OpenMP: enabled=1 max_threads=" << omp_get_max_threads()
                 << " places=" << num_places << " place_cpus=" << place_cpus << endl;
#else
      if (myrank == 0)
            cout << "==> ABE OpenMP: enabled=0 max_threads=1" << endl;
#endif

      double Begin_clock, End_clock;
      if (myrank == 0)
      {
            Begin_clock = MPI_Wtime();
      }

      if (argc > 1)
      {
            string sttr(argv[1]);
            parameters::str_par.insert(map<string, string>::value_type("inputpar", sttr));
      }
      else
      {
            string sttr("input.par");
            parameters::str_par.insert(map<string, string>::value_type("inputpar", sttr));
      }

      int checkrun;
      char checkfilename[50];
      int ID_type;
      int Steps;
      double StartTime, TotalTime;
      double AnasTime, DumpTime, d2DumpTime, CheckTime;
      double Courant;
      double numepss, numepsb, numepsh;
      int Symmetry;
      int a_lev, maxl, decn;
      double maxrex, drex;
      // read parameter from file
      {
            map<string, string>::iterator iter;
            string out_dir;
            const int LEN = 256;
            char pline[LEN];
            string str, sgrp, skey, sval;
            int sind;
            char pname[50];
            iter = parameters::str_par.find("inputpar");
            if (iter != parameters::str_par.end())
            {
                  out_dir = iter->second;
                  sprintf(pname, "%s", out_dir.c_str());
            }
            else
            {
                  cout << "Error inputpar" << endl;
                  exit(0);
            }
            ifstream inf(pname, ifstream::in);
            if (!inf.good() && myrank == 0)
            {
                  cout << "Can not open parameter file " << pname << endl;
                  MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int i = 1; inf.good(); i++)
            {
                  inf.getline(pline, LEN);
                  str = pline;

                  int status = misc::parse_parts(str, sgrp, skey, sval, sind);
                  if (status == -1)
                  {
                        cout << "error reading parameter file " << pname << " in line " << i << endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                  }
                  else if (status == 0)
                        continue;

                  if (sgrp == "ABE")
                  {
                        if (skey == "checkrun")
                              checkrun = atoi(sval.c_str());
                        else if (skey == "checkfile")
                              strcpy(checkfilename, sval.c_str());
                        else if (skey == "ID Type")
                              ID_type = atoi(sval.c_str());
                        else if (skey == "Steps")
                              Steps = atoi(sval.c_str());
                        else if (skey == "StartTime")
                              StartTime = atof(sval.c_str());
                        else if (skey == "TotalTime")
                              TotalTime = atof(sval.c_str());
                        else if (skey == "DumpTime")
                              DumpTime = atof(sval.c_str());
                        else if (skey == "d2DumpTime")
                              d2DumpTime = atof(sval.c_str());
                        else if (skey == "CheckTime")
                              CheckTime = atof(sval.c_str());
                        else if (skey == "AnalysisTime")
                              AnasTime = atof(sval.c_str());
                        else if (skey == "Courant")
                              Courant = atof(sval.c_str());
                        else if (skey == "Symmetry")
                              Symmetry = atoi(sval.c_str());
                        else if (skey == "small dissipation")
                              numepss = atof(sval.c_str());
                        else if (skey == "big dissipation")
                              numepsb = atof(sval.c_str());
                        else if (skey == "shell dissipation")
                              numepsh = atof(sval.c_str());
                        else if (skey == "Analysis Level")
                              a_lev = atoi(sval.c_str());
                        else if (skey == "Max mode l")
                              maxl = atoi(sval.c_str());
                        else if (skey == "detector number")
                              decn = atoi(sval.c_str());
                        else if (skey == "farest detector position")
                              maxrex = atof(sval.c_str());
                        else if (skey == "detector distance")
                              drex = atof(sval.c_str());
                        else if (skey == "output dir")
                              out_dir = sval;
                  }
            }
            inf.close();

            iter = parameters::str_par.find("output dir");
            if (iter != parameters::str_par.end())
            {
                  out_dir = iter->second;
            }
            else
            {
                  parameters::str_par.insert(map<string, string>::value_type("output dir", out_dir));
            }
      }

      if (myrank == 0)
      {
            string out_dir;
            char filename[50];
            map<string, string>::iterator iter;
            iter = parameters::str_par.find("output dir");
            if (iter != parameters::str_par.end())
            {
                  out_dir = iter->second;
            }
            sprintf(filename, "%s/setting.par", out_dir.c_str());
            ofstream setfile;
            setfile.open(filename, ios::trunc);

            if (!setfile.good())
            {
                  char cmd[100];
                  // sprintf(cmd,"rm %s -f",out_dir.c_str());
                  // system(cmd);
                  sprintf(cmd, "mkdir %s", out_dir.c_str());
                  system(cmd);

                  setfile.open(filename, ios::trunc);
            }

            time_t tnow;
            time(&tnow);
            struct tm *loc_time;
            loc_time = localtime(&tnow);
            setfile << "# File created on " << asctime(loc_time);
            setfile << "#" << endl;
            // echo the micro definition in "microdef.fh"
            setfile << "macro definition used in microdef.fh" << endl;

            setfile << "Frans' tetrad type for psi4 calculation" << endl;

            setfile << "Cell center numerical grid structure" << endl;

            setfile << "                   ghost zone = " << ghost_width << endl;

            setfile << "                  buffer zone = " << buffer_width << endl;

            setfile << "                  Gauge type = " << GAUGE << endl;

            setfile << "using BSSN variable for constraint violation and psi4 calculation" << endl;

            // echo the micro definition in "microdef.h"
            setfile << "macro definition used in microdef.h" << endl;
            setfile << "     Sommerfeld boundary type = " << SommerType << endl;
            setfile << "using Gauss integral in waveshell" << endl;
            setfile << "                     ABE type = " << ABEtype << endl;
            setfile << "                     ID  type = " << ID_type << endl;
            setfile << "        Psi4 calculation type = " << Psi4type << endl;
            setfile << "    RestrictProlong time type = " << RPS << endl;
            setfile << "    RestrictProlong scheme type = " << RPB << endl;
            setfile << "Enforce algebra constraint type = " << AGM << endl;
            setfile << "Analysis and PBH treat type = " << MAPBH << endl;
            setfile << "   mesh level parallel type = " << PSTR << endl;
            setfile << "                regrid type = " << REGLEV << endl;

            setfile << "                        dim = " << dim << endl;
            setfile << "               buffer_width = " << buffer_width << endl;
            setfile << "               SC_width = " << SC_width << endl;
            setfile << "               CS_width = " << CS_width << endl;

            setfile.close();
      }

      // echo parameters
      if (myrank == 0)
      {
            cout << endl;
            cout << " /////////////////////////////////////////////////////////////// " << endl;
            cout << " AMSS-NCKU Begin !!! " << endl;
            cout << " /////////////////////////////////////////////////////////////// " << endl;
            cout << endl;

            if (checkrun)
                  cout << "                             checked run" << endl;
            else
                  cout << "                                 new run" << endl;

            cout << "   simulation with cpu numbers = " << nprocs << endl;
            cout << "               simulation time = (" << StartTime << ", " << TotalTime << ")" << endl;
            cout << " simulation steps for this run = " << Steps << endl;
            cout << "                Courant number = " << Courant << endl;

            switch (ID_type)
            {
            case -3:
                  cout << "            Initial Data Type: Analytical NBH (Cao's Formula)" << endl;
                  break;
            case -2:
                  cout << "            Initial Data Type: Analytical Kerr-Schild" << endl;
                  break;
            case -1:
                  cout << "            Initial Data Type: Analytical NBH (Lousto's Formula)" << endl;
                  break;
            case 0:
                  cout << "            Initial Data Type: Numerical Ansorg TwoPuncture" << endl;
                  break;
            case 1:
                  cout << "            Initial Data Type: Numerical Pablo" << endl;
                  break;
            default:
                  cout << " OOOOps, not supported Initial Data setting!" << endl;
                  MPI_Abort(MPI_COMM_WORLD, 1);
            }

            switch (Symmetry)
            {
            case 0:
                  cout << "             Symmetry setting: No_Symmetry" << endl;
                  break;
            case 1:
                  cout << "             Symmetry setting: Equatorial" << endl;
                  break;
            case 2:
                  cout << "             Symmetry setting: Octant" << endl;
                  break;
            default:
                  cout << " OOOOps, not supported Symmetry setting!" << endl;
                  MPI_Abort(MPI_COMM_WORLD, 1);
            }

            cout << " Courant = " << Courant << endl;
            cout << " artificial dissipation for shell patches = " << numepsh << endl;
            cout << " artificial dissipation for fixed levels = " << numepsb << endl;
            cout << " artificial dissipation for moving levels = " << numepss << endl;
            cout << " Dumpt Time = " << DumpTime << endl;
            cout << " Check Time = " << CheckTime << endl;
            cout << " Analysis Time = " << AnasTime << endl;
            cout << " Analysis level = " << a_lev << endl;
            cout << " checkfile = " << checkfilename << endl;

            switch (ghost_width)
            {
            case 2:
                  cout << " second order finite difference is used" << endl;
                  break;
            case 3:
                  cout << " fourth order finite difference is used" << endl;
                  break;
            case 4:
                  cout << " sixth order finite difference is used" << endl;
                  break;
            case 5:
                  cout << " eighth order finite difference is used" << endl;
                  break;
            default:
                  cout << " Why are you using ghost width = " << ghost_width << endl;
                  MPI_Abort(MPI_COMM_WORLD, 1);
            }

            cout << "///////////////////////////////////////////////////////////////" << endl;
      }

      //===========================the computation body====================================================

      bssn_class *ADM;

      ADM = new bssn_class(Courant, StartTime, TotalTime, DumpTime, d2DumpTime, CheckTime, AnasTime,
                           Symmetry, checkrun, checkfilename, numepss, numepsb, numepsh,
                           a_lev, maxl, decn, maxrex, drex);

      ADM->Initialize();

      // new code   Xiao Qu
      switch (ID_type)
      {
      case (-3):
            // set up initial data with Cao's analytical formula
            ADM->Setup_Initial_Data_Cao();
            break;
      case (-2):
            // set up initial data with KerrSchild analytical formula
            ADM->Setup_KerrSchild();
            break;
      case (-1):
            // set up initial data with Lousto's analytical formula
            ADM->Setup_Initial_Data_Lousto();
            break;
      case (0):
            // set up initial data with Ansorg TwoPuncture Solver
            ADM->Read_Ansorg();
            break;
      case (1):
            // set up initial data with Pablo's Olliptic Solver
            ADM->Read_Pablo();
            // ADM->Write_Pablo();
            break;
      default:
            if (myrank == 0)
            {
                  cout << "not recognized ABE::InitialDataType = " << ID_type << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
      }

#ifdef USE_GPU
      ADM->move_to_gpu();
#endif

      End_clock = MPI_Wtime();
      if (myrank == 0)
      {
            cout << endl;
            cout << " Before Evolve, it takes " << MPI_Wtime() - Begin_clock << " seconds" << endl;
            cout << endl;
      }

      ADM->Evolve(Steps);

      if (myrank == 0)
      {
            cout << endl;
            cout << " Total Evolve Time: "  << MPI_Wtime() - End_clock   << " seconds!" << endl;
            cout << " Total Running Time: " << MPI_Wtime() - Begin_clock << " seconds!" << endl;
            cout << endl;
      }

      delete ADM;

      //=======================caculation done=============================================================

      if (myrank == 0)
      {
            cout << endl;
            cout << " =============================================================== " << endl;
            cout << " Simulation is successfully done!! " << endl;
            cout << " =============================================================== " << endl;
            cout << endl;
            cout << " This run used " << MPI_Wtime() - Begin_clock << " seconds! " << endl;
            cout << endl;
      }

      MPI_Finalize();

      exit(0);
}

//===================================================================================================
//===================================================================================================
