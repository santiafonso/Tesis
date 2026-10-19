///-------------------------------------------------------------------------------------///
///----------------------------------KMC galvanostatico
///CORREGIDO----------------------///
///-------------------------------------------------------------------------------------///

#include "acumulador.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <vector>

#ifdef TIMING
#include <chrono>
#endif

/// DEFINO CONSTANTES - MANTENIDAS EXACTAS
static constexpr int PASTEMP = 1;
static constexpr double NPTS = PASTEMP;
static constexpr int Nbins = 250;
static constexpr int NMUESTRAS = 1;
static constexpr int frames = 0;

/// Galvanostatic parameters inputs - AHORA VARIABLES
static double xi = -1; 
static double el = -1.147; 
static double Chi;
static double Ele;
static constexpr double dif_coeff = 3.5e-12;

/// Cell and energy parameters
static constexpr double Ncel_x = 40;
static constexpr double Ncel_y = 20;
static constexpr double Ncel_z = 40;
static constexpr double Nsitu = 1;
static constexpr int Npt = Nsitu * Ncel_x * Ncel_y * Ncel_z;
static constexpr double Np = Npt;
static constexpr double ac = 1.0;
static constexpr double Lx = ac * Ncel_x;
static constexpr double Ly = ac * Ncel_y;
static constexpr double Lz = ac * Ncel_z;
static constexpr int Nvec1 = 6;
static constexpr int Nvec2 = 0;
static constexpr int Nvec3 = 0;
static constexpr double rcorte1 = ac;
static constexpr double rcorte2 = 0.0;
static constexpr double rcorte3 = 0.0;
static constexpr int Nvec = 6;
static constexpr int SinVec = 1;
static constexpr int Nvecij = 10;
static constexpr double Area = Lx * Lz * 1.0e-16;
static constexpr int plano = Nsitu * Ncel_x * Ncel_z;
static constexpr double ce = 1.60217663e-19;
static constexpr double Qmax = Np * ce;
static constexpr double BK = 0.00008617385 * 1000.0;
static constexpr double T = 298.0;
static constexpr double kT = BK * T;
static constexpr double g_pot = 0.0; // Parametro de Frumkin
static constexpr double J1 = (g_pot / 6.0) * kT; //      ///meV 28.1425;
double J2 = 0.0;

/// Galvanostatic constants - AHORA CALCULADAS DINÁMICAMENTE
double Crate;
double ic;
double it;
double k0;
const constexpr double ds = 1e-8 * ac;
const constexpr double E0 = 0.0;

/// Kinetic parameters
static constexpr double doskT = 2.0 * BK * T;
static constexpr double kdif = 1.0e13;
static constexpr double kads = 1.0e13;
static constexpr double kdes = 1.0e13;
static constexpr double Height_dif =
    -log(dif_coeff / (kdif * ac * ac * 1e-16)) * kT;
static double Height_k0; // AHORA VARIABLE
static constexpr double Eadif = Height_dif / kT;
static double Eaads; // AHORA VARIABLE
static constexpr int Nevento = 8;
static constexpr int NTHREADS = 32;

/// Galvanostatic - AHORA CALCULADAS DINÁMICAMENTE
static double dt;
static double timestep;
static double printt;
static double pasobin;
static constexpr double muoff = 0.149 * 1000.0;
static double mui;

#define lee_info_en "CS-40x20x40.xyz"
static std::string grabadif_en; // AHORA VARIABLE
#define grabared_en "24x108.xyz"
#define grabavmd_en "vmd-10x40x10-100-2.xyz"
#define grabaver_en "ver-10x40x10-100-2.dat"
#define grabaparam_en "parametros-12.dat"

#define ULONGMAX 4294917238.0

// GENERADOR RANDOM ORIGINAL - MANTENIDO EXACTO
static unsigned int seed[256];
static unsigned int r;
static unsigned char irr;

inline double randomm(void) {
  return (double)(r = seed[irr++] += seed[r >> 24]) / ULONGMAX;
}

inline double randomm_01abierto(void) {
  double result;
  do {
    result = randomm();
  } while ((result <= 0.0) || (result >= 1.0));
  return result;
}

/// VARIABLES GLOBALES - ESTRUCTURA ORIGINAL MANTENIDA
static int j, n;
static int Na, Nd, cont2, numvec[Npt][Nevento], numvec2[Npt][Nvec], Nconst,
    Npart, numven[Npt][Nvec1], numven2[Npt][Nvec2], N1[Npt], N2[Npt], N3[Npt],
    ttt;
static int numven3[Npt][Nvec3], surface[plano], bottom[plano],
    bulk[Npt - 2 * plano];
static std::vector<int> NNi_data;
static double dx, dy, dz, CORD[Npt][3], Evento[Npt][Nevento], ti, HI, R22;
static double Tiempo, Ft, Energia[Npt][Nevento];
static double TiempoBin[Nbins + 1][2], En, Timetotal, tita2, tita[Nbins],
    TiempoMedio[Nbins];
static double Nint, Ndint, Ndif, En2, sumaI, It, corriente, CB, mayor, menor,
    MU, counter, nt, TIME, MUi, MUs[Nbins];
static int Ocup[Npt];

/// DECLARACIONES DE FUNCIONES
void Caja();
void CajaC();
static void Proceso(Acumulador<double> &acumulador);
void Vecinos();
static void Velocidades();
static void VelocidadesAds(int ii);
static void VelocidadesDif(int ii, int kk);
void Inicializa_generador();
void Graba();
void Vmd();
void Inicial(double xi_val, double el_val);
void Inicial2();
void Difusion();
void GrabaDif(int i, double xi_val, double el_val);
void GrabaVer();
void Carga();
void potencial();
void CalcularParametros(); // NUEVA FUNCIÓN

static void Actualizar(int i, int m, double Esitio);
static double CalcularEsitio(int i);
static double CalcularEf(int i, int j);

// Helpers de I/O - ORIGINALES
static FILE *AbrirArchivo(const char *path, const char *modo) {
  FILE *rv = fopen(path, modo);
  if (rv == nullptr) {
    fprintf(stderr, "Error abriendo archivo %s: %s\n", path, strerror(errno));
    exit(1);
  }
  return rv;
}

template <typename T>
static void LeerUnDato(FILE *archivo, const char *fmt, T *salida) {
  int rv = fscanf(archivo, fmt, salida);
  if (rv < 1) {
    fprintf(stderr, "Error leyendo valor formato %s.\n", fmt);
    exit(1);
  }
}

int main(int argc, char *argv[]) {

  if (argc < 3) {
    std::cerr << "Uso: " << argv[0] << " <valor_xi> <valor_el>" << std::endl;
    std::cerr << "Ejemplo: " << argv[0] << " -1.0 -1.147" << std::endl;
    return 1; // Salir con error si no se proporcionan ambos valores
  }

  // Leer valores de xi y el desde argumentos de línea de comandos
  xi = std::stod(argv[1]);
  el = std::stod(argv[2]);
  
  // Calcular parámetros dependientes
  CalcularParametros();

  /// Escribir hoja de parametros
  FILE *archivo = AbrirArchivo(grabaparam_en, "a");
  fprintf(archivo,
          "Chi=%f \nEle=%f \ng=%f \nNcel_x=%f \nNcel_y%f \nNcel_z=%f "
          "\nmuoff=%f \nNTHREADS=%d \n",
          (float)(xi), (float)(el), (float)(g_pot), (float)(Ncel_x),
          (float)(Ncel_y), (float)(Ncel_z), (float)(muoff), (int)(NTHREADS));
  fclose(archivo);

  Caja();
  Vecinos();
  Inicial(xi, el);

  auto acumulador = Acumulador<double>(Npt * Nevento);
  omp_set_num_threads(NTHREADS);

  /// PROCESAMIENTO
#ifdef TIMING
  auto start = std::chrono::steady_clock::now();
#endif

  for (int NM = 0; NM < NMUESTRAS; NM++) {
    Inicial2();
    int Kk = 0;
    double nn = 1.0;
    double nt = 1.0;
    Tiempo = 0.0;
    ttt = 0;
    counter = 0.0;
    MUi = 0.0;
    mui = MU = muoff - 1.0;

    while (mui < muoff) {
      Kk++;
      potencial();
      Proceso(acumulador);

#ifdef TIMING
      if (Kk % 1000 == 0) {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        // printf("%lf iterations per second\n", 1000 / diff.count());
        start = end;
      }
#endif

      if ((Tiempo - nt * timestep) > timestep) {
        MU = MUi / counter;
        // printf("point =  %f\n",MU/1000);
        nt++;
        counter = MUi = 0.0;
      }
    }
  }

  /// OUTPUT
  for (int i = 0; i < Nbins; i++) {
    MUs[i] /= TiempoBin[i][1];
    tita[i] /= TiempoBin[i][1];
    if (TiempoBin[i][1] == 0) {
      tita[i] = 0.0;
    }

    GrabaDif(i, xi, el);
  }

  return 0;
}

/// NUEVA FUNCIÓN PARA CALCULAR PARÁMETROS DEPENDIENTES
void CalcularParametros() {
  Chi = pow(10, xi);
  Ele = pow(10, el);
  
  Crate = dif_coeff * Ele * 3600 / (Ly * Ly * 1e-16);
  ic = -Crate * Qmax / (Area * 3.6);
  it = ic * Area / (1000.0 * ce);
  k0 = Chi * sqrt(dif_coeff * Crate / 3600);
  
  Height_k0 = -log(k0 / (kads * ds)) * kT;
  Eaads = Height_k0 / kT;
  
  dt = abs(1000.0 * Qmax / (ic * Area));
  timestep = dt / Nbins;
  printt = timestep;
  pasobin = dt / Nbins;
  
  // Crear nombre de archivo con xi y el
  char buffer[256];
  snprintf(buffer, sizeof(buffer), "datos-40x40x40-Chi-%.3f-El-%.3f-g0", xi, el);
  grabadif_en = std::string(buffer);
}

/// PROCESO - LÓGICA ORIGINAL EXACTA
static void Proceso(Acumulador<double> &acumulador) {
  acumulador.Preparar(&Evento[0][0]);
  double sumaV = acumulador.TotalAcumulado();

  /// EVENT SELECTION - LÓGICA ORIGINAL EXACTA
  double R;
  int indice = -1;
  do {
    R = randomm_01abierto() * sumaV;
    indice = acumulador.Buscar(R);
  } while (indice < 0);

  int ii = indice / Nevento;
  int kk = indice % Nevento;

  /// Tiempo - EXACTO
  double Rdos = randomm_01abierto();
  ti = -log(Rdos) / sumaV;
  Tiempo += ti;

  /// EVENT REALIZATION - EXACTO
  switch (kk) {
  case Nevento - 2: {
    Ocup[ii] = true;
    Nconst++;
    Nint++;
  } break;
  case Nevento - 1: {
    Ocup[ii] = false;
    Nconst--;
    Ndint++;
  } break;
  default: {
    Ocup[ii] = false;
    Ocup[numvec[ii][kk]] = true;
    Ndif++;
  } break;
  }

  // BINNING - EXACTO
  int dd = 0;
  while (dd < Nbins) {
    if ((Tiempo >= TiempoBin[dd][0]) && (Tiempo < TiempoBin[dd + 1][0])) {
      TiempoBin[dd][1]++;
      MUs[dd] += mui;
      tita[dd] += Nconst;
      dd = Nbins + 1;
    } else {
      dd++;
    }
  }
  MUi += mui;

  if (kk == Nevento - 2) {
    Na++;
  }
  if (kk == Nevento - 1) {
    Nd++;
  }
  counter++;

  /// VECTORS ACTUALIZATION - EXACTO
  sumaI = 0.0;
  switch (kk) {
  case Nevento - 2: {
    VelocidadesAds(ii);
    for (int i = 0; i < plano; i++) {
      sumaI -= Evento[i][Nevento - 2];
    }
  } break;
  case Nevento - 1: {
    VelocidadesAds(ii);
    for (int i = 0; i < plano; i++) {
      sumaI += Evento[i][Nevento - 1];
    }
  } break;
  default: {
    VelocidadesDif(ii, kk);
  } break;
  }
}

void Inicial(double xi_val, double el_val) {
  double a;

  Tiempo = ti = sumaI = corriente = HI = Npart = R22 = 0.0;

  Inicializa_generador();
  Na = Nd = 0;

  std::string nombreArchivo = grabadif_en + ".dat";
  FILE *archivo = AbrirArchivo(nombreArchivo.c_str(), "a");
  fprintf(archivo, "SoC E[V] Tiempo logChi logEle\n");
  fclose(archivo);

  a = 0.0;
  for (int i = 0; i <= Nbins; i++) {
    TiempoBin[i][0] = a;
    a += pasobin;
  }
  for (int i = 0; i < Nbins; i++) {
    tita[i] = TiempoMedio[i] = MUs[i] = 0.0;
  }
  for (int i = 0; i < Nbins; i++) {
    TiempoMedio[i] =
        TiempoBin[i][0] + ((TiempoBin[i + 1][0] - TiempoBin[i][0]) / 2);
  }
}

void Inicial2() {
  Tiempo = ti = sumaI = corriente = Npart = R22 = Nconst = HI = 0.0;

  for (int i = 0; i < Npt; i++) {
    Ocup[i] = false;
  }
  for (int i = 0; i < Npt; i++) {
    for (int j = 0; j < Nevento; j++) {
      Evento[i][j] = Energia[i][j] = 0.0;
    }
  }

  CB = kads * exp(-Eaads);
  Velocidades();
}

/// POTENCIAL - EXACTO
void potencial() {
  int Nit, n1, NI, NR;
  double CA, muui, CC, CD, CE, CF, hola, CG, CH, Esitio, Ef, Error, ErrorIt, a,
      b, bb, c, d, fa, fb, fc, s, fs, fbb, x0, xi, fx, dfx, div, fx1, div1, x1,
      xinf, xsup, fxi, divi;
  int mflag, ij;
  CD = CC = 0.0;

  for (int i = 0; i < plano; i++) {
    Esitio = Ef = 0.0;
    ij = surface[i];
    switch (Ocup[ij]) {
    case 1: { /// Oxidacion
      for (n = 0; n < N1[ij]; n++) {
        j = numven[ij][n];
        if (Ocup[j] == 1) {
          Esitio += J1;
        }
      }
      Esitio += E0;
      CC += exp(Esitio / doskT); /// Esitio=Ei
    } break;
    case 0: { /// Reduccion
      for (n = 0; n < N1[ij]; n++) {
        j = numven[ij][n];
        if (Ocup[j] == 1) {
          Esitio += J1;
        }
      }
      Esitio += E0;
      CD += exp(-Esitio / doskT); /// Esitio=Ef
    } break;
    }
  }

  CE = CB * CC;
  CF = CB * CD;

  /// METODO DE BREN - EXACTO
  muui = mui;
  hola = 0.0;
  Error = 1.0e-10;
  ErrorIt = Error + 1.0;
VV:
  a = -2500.0 - hola;
  b = 2000.0 + hola;
  fa = it - (CE * exp(-a / doskT) - CF * exp(a / doskT));
  fb = it - (CE * exp(-b / doskT) - CF * exp(b / doskT));
  if ((fa * fb) >= 0.0) {
    hola++;
    goto VV;
  }
  if (fabs(fa) < fabs(fb)) {
    bb = b;
    b = a;
    a = bb;
    fbb = fb;
    fb = fa;
    fa = fbb;
  }
  c = a;
  fc = fa;
  fs = 2.0;
  s = 0.0;
  d = 0.0;
  mflag = 1;

  while (ErrorIt > Error) {

    if (fa != fc && fb != fc) {
      s = (a * fb * fc / ((fa - fb) * (fa - fc))) +
          (b * fa * fc / ((fb - fa) * (fb - fc))) +
          (c * fa * fb / ((fc - fa) * (fc - fb))); /// IQI
    } else {
      s = b - (fb * (b - a) / (fb - fa)); /// SECANTE
    }

    if (((s < ((3.0 * (a + b)) * 0.25)) || (s > b)) ||
        (mflag == 1 && (fabs(s - b) >= (fabs(b - c) * 0.5))) ||
        (mflag == 0 && (fabs(s - b) >= (fabs(c - d) * 0.5))) ||
        (mflag == 1 && (fabs(b - c) < Error)) ||
        (mflag == 0 && (fabs(c - d) < Error))) {
      /// BISECCION
      s = (a + b) * 0.5;
      mflag = 1;
    } else {
      mflag = 0;
    }

    fs = it - (CE * exp(-s / doskT) - CF * exp(s / doskT)); /// calculate fs
    d = c;
    c = b;
    fc = fb;

    if ((fa * fs) < 0.0) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }

    if (fabs(fa) < fabs(fb)) {
      bb = b;
      b = a;
      a = bb;
      fbb = fb;
      fb = fa;
      fa = fbb;
    }
    ErrorIt = fabs(b - a);

  } /// END WHILE
  mui = s;
}

void Vecinos() {
  int n, n1, n2, n3, n4;
  double r2, dx2, dy2;
  int Nvec = 6;
  mayor = 0.0;
  menor = 1e5;

  for (int i = 0; i < Npt; i++) {
    /// Encontrar mayor y menor valor del eje Y (eje de difusion)
    if (CORD[i][1] > mayor)
      mayor = CORD[i][1];
    if (CORD[i][1] < menor)
      menor = CORD[i][1];

    for (int j = 0; j < Nvec; j++) {
      numvec[i][j] = -1;
    }
  }

  /// Put surface, bottom and bulk sites in vectors
  n = n1 = n2 = 0;
  for (int i = 0; i < Npt; i++) {
    if (CORD[i][1] < menor + 0.1) {
      surface[n] = i;
      n++;
    }
    if (CORD[i][1] > mayor - 0.1) {
      bottom[n1] = i;
      n1++;
    }
    if ((CORD[i][1] < mayor) && (CORD[i][1] > menor)) {
      bulk[n2] = i;
      n2++;
    }
  }

  n = n1 = n2 = 0;
  int nM = 6;
  /// PRIMER PLANO
  for (int i = 0; i < plano; i++) {
    n = 0;
    N1[surface[i]] = 0; // INICIALIZAR CONTADOR
    for (int j = 0; j < Npt; j++) {
      dx = fabs(CORD[surface[i]][0] - CORD[j][0]);
      dy = fabs(CORD[surface[i]][1] - CORD[j][1]);
      dz = fabs(CORD[surface[i]][2] - CORD[j][2]);
      if (dz > 0.5 * Lz) {
        dz = Lz - dz;
      }
      if (dx > 0.5 * Lx) {
        dx = Lx - dx;
      }

      double radio = sqrt((dx * dx) + (dy * dy) + (dz * dz));
      if (surface[i] != j) {
        if (radio < rcorte1 + 0.1) {
          numvec[surface[i]][n] = j;
          numven[surface[i]][N1[surface[i]]] = j; // CORREGIR INDEXING
          n++;
          N1[surface[i]]++;
        }
      }
    }
  }

  /// ULTIMO PLANO
  for (int i = 0; i < plano; i++) {
    n = 0;
    N1[bottom[i]] = 0; // INICIALIZAR CONTADOR
    for (int j = 0; j < Npt; j++) {
      dx = fabs(CORD[bottom[i]][0] - CORD[j][0]);
      dy = fabs(CORD[bottom[i]][1] - CORD[j][1]);
      dz = fabs(CORD[bottom[i]][2] - CORD[j][2]);
      if (dz > 0.5 * Lz) {
        dz = Lz - dz;
      }
      if (dx > 0.5 * Lx) {
        dx = Lx - dx;
      }
      double radio = sqrt((dx * dx) + (dy * dy) + (dz * dz));
      if (bottom[i] != j) {
        if (radio < rcorte1 + 0.1) {
          numvec[bottom[i]][n] = j;
          numven[bottom[i]][N1[bottom[i]]] = j; // CORREGIR INDEXING
          n++;
          N1[bottom[i]]++;
        }
      }
    }
  }

  /// RESTO DE LA RED (4)
  for (int i = 0; i < Npt - 2 * plano; i++) {
    n = 0;
    N1[bulk[i]] = 0; // INICIALIZAR CONTADOR
    for (int j = 0; j < Npt; j++) {
      dx = fabs(CORD[bulk[i]][0] - CORD[j][0]);
      dy = fabs(CORD[bulk[i]][1] - CORD[j][1]);
      dz = fabs(CORD[bulk[i]][2] - CORD[j][2]);
      if (dz > 0.5 * Lz) {
        dz = Lz - dz;
      }
      if (dx > 0.5 * Lx) {
        dx = Lx - dx;
      }
      double radio = sqrt((dx * dx) + (dy * dy) + (dz * dz));
      if (bulk[i] != j) {
        if (radio < rcorte1 + 0.1) {
          numvec[bulk[i]][n] = j;
          numven[bulk[i]][N1[bulk[i]]] = j; // CORREGIR INDEXING
          n++;
          N1[bulk[i]]++;
        }
      }
    }
  }
}

/// VELOCIDADES - LÓGICA ORIGINAL CON PARALELIZACIÓN CONSERVATIVA
static void Velocidades() {
  double sumaI = 0.0;

// Inicializar en paralelo pero manteniendo la lógica original
#pragma omp parallel for schedule(static)
  for (int i = 0; i < Npt; i++) {
    for (int j = 0; j < Nevento; j++) {
      Evento[i][j] = Energia[i][j] = 0.0;
    }
  }

  // Procesamiento secuencial para mantener exactitud
  /// Primer plano
  for (int i = 0; i < plano; i++) {
    double Esitio = CalcularEsitio(surface[i]);
    if (Ocup[surface[i]]) {
      for (int jj = 0; jj < Nvec - SinVec; jj++) {
        Actualizar(surface[i], jj, Esitio);
      }
      Energia[surface[i]][Nevento - 1] = -(Esitio + E0);
      Evento[surface[i]][Nevento - 1] =
          kads *
          exp(-(Eaads) - ((Energia[surface[i]][Nevento - 1] + mui) / (doskT)));
      sumaI += Evento[surface[i]][Nevento - 1];

    } else {
      Energia[surface[i]][Nevento - 2] = Esitio + E0;
      Evento[surface[i]][Nevento - 2] =
          kads *
          exp(-(Eaads) - ((Energia[surface[i]][Nevento - 2] - mui) / (doskT)));
      sumaI += Evento[surface[i]][Nevento - 2];
    }
  }

  /// Ultimo plano
  for (int i = 0; i < plano; i++) {
    double Esitio = CalcularEsitio(bottom[i]);
    if (Ocup[bottom[i]]) {
      for (int jj = 0; jj < Nvec - SinVec; jj++) {
        Actualizar(bottom[i], jj, Esitio);
      }
    }
  }

  /// Resto
  for (int i = 0; i < Npt - 2 * plano; i++) {
    double Esitio = CalcularEsitio(bulk[i]);
    if (Ocup[bulk[i]]) {
      for (int jj = 0; jj < Nvec; jj++) {
        Actualizar(bulk[i], jj, Esitio);
      }
    }
  }
}

/// VELOCIDADES ADS - MANTENIDO EXACTO CON PARALELIZACIÓN MÍNIMA
static void VelocidadesAds(int ii) {
  int i = ii;
  /// Primer plano - actualizar sitio principal
  std::fill_n(Evento[i], Nevento, 0.0);
  std::fill_n(Energia[i], Nevento, 0.0);
  double Esitio = CalcularEsitio(i);
  if (Ocup[i]) {
    for (int jj = 0; jj < Nvec - SinVec; jj++) {
      Actualizar(i, jj, Esitio);
    }
    Energia[i][Nevento - 1] = -(Esitio + E0);
    Evento[i][Nevento - 1] =
        kads * exp(-(Eaads) - ((Energia[i][Nevento - 1] + mui) / (doskT)));
  } else {
    Energia[i][Nevento - 2] = Esitio + E0;
    Evento[i][Nevento - 2] =
        kads * exp(-(Eaads) - ((Energia[i][Nevento - 2] - mui) / doskT));
  }

  // Actualizar vecinos secuencialmente para mantener exactitud
  for (int jj = 0; jj < Nvec - SinVec; jj++) {
    int i = numvec[ii][jj]; // Todos los vecinos de ii
    std::fill_n(Evento[i], Nevento, 0.0);
    std::fill_n(Energia[i], Nevento, 0.0);
    /// Primer plano
    if (CORD[i][1] < menor + 0.1) {
      double Esitio = CalcularEsitio(i);
      if (Ocup[i]) {
        for (int gg = 0; gg < Nvec - SinVec; gg++) {
          Actualizar(i, gg, Esitio);
        }

        Energia[i][Nevento - 1] = -(Esitio + E0);
        Evento[i][Nevento - 1] =
            kads * exp(-(Eaads) - ((Energia[i][Nevento - 1] + mui) / (doskT)));
      } else {
        Energia[i][Nevento - 2] = Esitio + E0;
        Evento[i][Nevento - 2] =
            kads * exp(-(Eaads) - ((Energia[i][Nevento - 2] - mui) / (doskT)));
      }
    } /// PRIMER PLANO
    /// Resto
    else {
      if (Ocup[i]) {
        double Esitio = CalcularEsitio(i);
        for (int gg = 0; gg < Nvec; gg++) {
          Actualizar(i, gg, Esitio);
        }
      }
    } /// DE RESTO
  }   /// NXY
}

/// VELOCIDADES DIF - MANTENIDO EXACTO
static void VelocidadesDif(int ii, int kk) {
  int i = ii;
  double Esitio = CalcularEsitio(i);
  /// ACTUALIZO LAS VELOCIDADES DEL SITIO DE DONDE SE FUE LA PARTICULA

  std::fill_n(Evento[i], Nevento, 0.0);
  std::fill_n(Energia[i], Nevento, 0.0);

  /// Primer plano
  if (CORD[i][1] < menor + 0.1) {
    Energia[i][Nevento - 2] = Esitio + E0;
    Evento[i][Nevento - 2] =
        kads * exp(-(Eaads) - ((Energia[i][Nevento - 2] - mui) / (doskT)));
  } ///

  // Actualizar vecinos de ii
  for (int jji = 0; jji < Nvec; jji++) {
    int iii = numvec[ii][jji];
    if (iii != -1) {
      std::fill_n(Evento[iii], Nevento, 0.0);
      std::fill_n(Energia[iii], Nevento, 0.0);
      double Esitio = CalcularEsitio(iii);

      /// Primer plano
      if (CORD[iii][1] < menor + 0.1) {
        if (Ocup[iii]) {
          for (int gg = 0; gg < Nvec - SinVec; gg++) {
            Actualizar(iii, gg, Esitio);
          }
          Energia[iii][Nevento - 1] = -(Esitio + E0);
          Evento[iii][Nevento - 1] =
              kads *
              exp(-(Eaads) - ((Energia[iii][Nevento - 1] + mui) / (doskT)));
        } else {
          Energia[iii][Nevento - 2] = Esitio + E0;
          Evento[iii][Nevento - 2] =
              kads *
              exp(-(Eaads) - ((Energia[iii][Nevento - 2] - mui) / (doskT)));
        }
        /// Resto
      } else if (CORD[iii][1] > mayor - 0.1) {
        if (Ocup[iii]) {
          for (int gg = 0; gg < Nvec - SinVec; gg++) {
            Actualizar(iii, gg, Esitio);
          }
        }
      } else {
        if (Ocup[iii]) {
          double Esitio = CalcularEsitio(iii);
          for (int gg = 0; gg < Nvec; gg++) {
            Actualizar(iii, gg, Esitio);
          }
        }
      }
    }
  }

  // Actualizar vecinos de destino (numvec[ii][kk])
  for (int ijj = 0; ijj < Nvec; ijj++) {
    int iiii = numvec[numvec[ii][kk]][ijj];
    if (iiii != -1) {
      double Esitio = CalcularEsitio(iiii);
      std::fill_n(Evento[iiii], Nevento, 0.0);
      std::fill_n(Energia[iiii], Nevento, 0.0);

      /// Primer plano
      if (CORD[iiii][1] < menor + 0.1) {
        if (Ocup[iiii]) {
          for (int ggg = 0; ggg < Nvec - SinVec; ggg++) {
            Actualizar(iiii, ggg, Esitio);
          }
          Energia[iiii][Nevento - 1] = -(Esitio + E0);
          Evento[iiii][Nevento - 1] =
              kads *
              exp(-(Eaads) - ((Energia[iiii][Nevento - 1] + mui) / (doskT)));
        } else {
          Energia[iiii][Nevento - 2] = Esitio + E0;
          Evento[iiii][Nevento - 2] =
              kads *
              exp(-(Eaads) - ((Energia[iiii][Nevento - 2] - mui) / (doskT)));
        }
        /// ULTIMO
      } else if (CORD[iiii][1] > mayor - 0.1) {
        if (Ocup[iiii]) {
          for (int ggg = 0; ggg < Nvec - SinVec; ggg++) {
            Actualizar(iiii, ggg, Esitio);
          }
        }
      } else {
        if (Ocup[iiii]) {
          double Esitio = CalcularEsitio(iiii);
          for (int ggg = 0; ggg < Nvec; ggg++) {
            Actualizar(iiii, ggg, Esitio);
          }
        }
      }
    }
  }
}

/// FUNCIONES DE AYUDA - EXACTAS DEL ORIGINAL
static double CalcularEsitio(int i) {
  double Esitio = 0.0;
  for (int n = 0; n < N1[i]; n++) {
    int j = numven[i][n];
    if (Ocup[j]) {
      Esitio += J1;
    }
  }
  return Esitio;
}

static double CalcularEf(int i, int j) {
  double Ef = 0.0;
  for (int n = 0; n < N1[j]; n++) {
    int k = numven[j][n];
    if (Ocup[k]) {
      Ef += J1;
    }
  }
  return Ef;
}

static void Actualizar(int i, int m, double Esitio) {
  int j = numvec[i][m];
  if (!Ocup[j]) {
    Energia[i][m] = CalcularEf(i, j) - Esitio;
    Evento[i][m] = kdif * exp(-(Energia[i][m] / (doskT)) - (Eadif));
  }
}

void GrabaDif(int i, double xi_val, double el_val) {

  std::string nombreArchivo = grabadif_en + ".dat";
  FILE *archivo = AbrirArchivo(nombreArchivo.c_str(), "a");
  fprintf(archivo, "%f %f %f %f %f\n", (tita[i] / (Npt)),
          (float)(-MUs[i] * 1.0e-3), (float)(TiempoMedio[i]), xi_val, el_val);
  fclose(archivo);
}

void GrabaVer() {
  float NNN = Nconst;
  FILE *archivo = AbrirArchivo(grabaver_en, "a");
  fprintf(archivo, "%f %f %f %f %f %f %f %f %f %f %f %d %d %d \n",
          (float)(Tiempo), (float)(-mui * 1e-3), (float)(NNN / Npt),
          (float)(ic), (float)(Nconst), (float)(Nint), (float)(Ndint),
          (float)(Ndif), (float)(HI), (float)(Eadif * kT), (float)(Eaads * kT),
          (int)(Lx * ac / 5.0), (int)(Ly * ac / 5.0), (int)(Npt * ac / 5.0));
  fclose(archivo);
}

void Inicializa_generador(void) {
#ifndef FIXEDSEED
  // inicializo generador pseudorandom: semilla = segundos desde 1970
  srand((unsigned)time(0));
#else
  // semilla fija para resultados deterministas
  srand(1);
#endif

  irr = 1;

  for (int i = 0; i < 256; ++i) {
    seed[i] = rand();
  }
  r = seed[0];

  for (int i = 0; i < 70000; ++i) {
    r = seed[irr++] += seed[r >> 24];
  }
}

void Vmd() {
  FILE *archivo1 = AbrirArchivo(grabavmd_en, "a");
  fprintf(archivo1, "%d \n", (int)(Npt));
  fprintf(archivo1, "\n");
  for (int i = 0; i < Npt; i++) {
    if (Ocup[i]) {
      fprintf(archivo1, "Li %4.5f %4.5f %4.5f\n", (float)(CORD[i][0]),
              (float)(CORD[i][1]), (float)(CORD[i][2]));
    } else {
      fprintf(archivo1, "Li %4.5f %4.5f -1.0\n", (float)(Lx / 2),
              (float)(Ly / 2));
    }
  }
  fclose(archivo1);
}

void Carga() {
  int e1;
  int E;
  double B, C, D;
  char Li[4];

  FILE *archivo1 = AbrirArchivo(lee_info_en, "r");

  for (int jj = 0; jj < frames; jj++) {
    LeerUnDato(archivo1, "%d", &E);
    fscanf(archivo1, "\n");
    for (int ii = 0; ii < Npt; ii++) {
      LeerUnDato(archivo1, "%3s", Li);
      LeerUnDato(archivo1, "%le", &B);
      LeerUnDato(archivo1, "%le", &C);
      LeerUnDato(archivo1, "%le", &D);
    }
  }
  e1 = 0;
  LeerUnDato(archivo1, "%d", &E);
  fscanf(archivo1, "\n");
  for (int ii = 0; ii < Npt; ii++) {
    LeerUnDato(archivo1, "%3s", Li);
    LeerUnDato(archivo1, "%le", &B);
    LeerUnDato(archivo1, "%le", &C);
    LeerUnDato(archivo1, "%le", &D);
    if (D < 0.0) {
      Ocup[ii] = false;
    } else {
      Ocup[ii] = true;
      e1++;
    }
  }
  fclose(archivo1);

  Nconst = e1;
}

void Caja() {
  int e1, jj, ii;
  int E;
  double B, C, D;
  char Li;
  FILE *archivo1 = AbrirArchivo(lee_info_en, "r");

  fscanf(archivo1, "%d", &E);
  fscanf(archivo1, "\n");
  for (ii = 0; ii < Npt; ii++) {
    fscanf(archivo1, "%s", &Li);
    fscanf(archivo1, "%le", &B);
    CORD[ii][0] = B;
    fscanf(archivo1, "%le", &C);
    CORD[ii][1] = C;
    fscanf(archivo1, "%le", &D);
    CORD[ii][2] = D;
  }

  fclose(archivo1);
}

// Funciones dummy para compatibilidad
void CajaC() {}
void Graba() {}
void Difusion() {}
void GrabaDif(int i) {}