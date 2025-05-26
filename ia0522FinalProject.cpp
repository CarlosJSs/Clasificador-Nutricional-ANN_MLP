#include <iostream>
#include <fstream>               // Lectura/escritura de archivos
#include <sstream>               // Manipulacion de strings (stringstream)
#include <string>
#include <vector>
#include <map>                   // std::map para almacenar valores nutrimentales
#include <regex>                 // Expresiones regulares para limpiar texto
#include <algorithm>             // std::transform, std::find, etc.

#include <opencv2/core.hpp>      // Nucleo de OpenCV (Mat, Size, etc.)
#include <opencv2/highgui.hpp>   // Cargar/mostrar imagenes
#include <opencv2/imgproc.hpp>   // Procesamiento de imagenes (cvtColor, threshold, etc.)
#include <opencv2/ml.hpp>        // Modulo de machine learning (ANN_MLP)
#include <opencv2/opencv.hpp>    // Acceso global a todo OpenCV

#include <tesseract/baseapi.h>   // API de Tesseract OCR
#include <leptonica/allheaders.h>// Libreria base que usa Tesseract

#include <locale.h>              // setlocale para caracteres especiales
#include <windows.h>             // SetConsoleOutputCP para UTF-8 en la consola de Windows


using namespace std;
using namespace cv;
using namespace cv::ml;

// Funciones reutilizadas del profesor yañez
//      Para el uso y creacion de la ANN_MLP
int syFeatureAndLabelMatrix_Read(string filename, Mat &fullFeatureMat);
int syImShuffleRows(const Mat &src, Mat &dst);
int syFeatureAndLabelMatrix_Split(const Mat &fullFeatureMat, Mat &trainMat, Mat &trainLabelsMat, Mat &testMat, Mat &testLabelsMat, int nFolds);
int syANN_MLP_Train_and_Test(int nClasses, const Mat &trainMat, const Mat &trainLabelsMat, const Mat &testMat, const Mat &testLabelsMat, Mat &confusion);

// Funciones para el OCR
map<string, float> extraerValoresNutricionales(const string& rutaImagen);
string limpiarTextoOCR(const string& texto);

// Menu principal
int main(){
    // Para los caracteres especiales
    setlocale(LC_ALL, "");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    int opcion;
    string modelo = "modelo_final.yml";

    do {
        cout << "\n\n\t- - - - - - - -   M E N U   - - - - - - - -" << endl;
        cout << "\n\t1) Obtener caracteristicas de una serie de imagenes\n";
        cout << "\t2) Entrenar red neuronal con archivo CSV\n";
        cout << "\t3) Probar una muestra individual (imagen)\n";
        cout << "\t4) Probar una muestra individual (.csv)\n";
        cout << "\t0) Salir\n\n\t\tSeleccione una opcion: ";
        cin >> opcion;

        switch(opcion){
            case 1: {
                string carpetaBase = "dbCategorias";
                string archivoSalida = "dataset_generado.csv";

                // Intentamos crear el csv de caracteristicas
                ofstream archivoCSV(archivoSalida);
                if (!archivoCSV.is_open()) {
                    cout << "Error al crear el archivo CSV de salida.\n";
                    continue;
                }

                // Escribir encabezados en el csv
                archivoCSV << "proteina,grasas_totales,grasas_trans,hidratos_disp,azucares,azucares_añadidos,fibra,sodio,categoria\n";

                vector<string> extensiones = { ".jpg", ".png", ".jpeg", ".bmp" };

                // Recorremos las carpetas de las categorias (0 - 4)
                for (int cat = 0; cat <= 4; cat++) {
                    string rutaCategoria = carpetaBase + "/" + to_string(cat);
                    vector<String> archivos;
                    glob(rutaCategoria + "/*", archivos, false);

                    for (const auto& rutaImg : archivos) {
                        // Verificamos extension de las imagenes
                        string ext = rutaImg.substr(rutaImg.find_last_of('.'));
                        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (find(extensiones.begin(), extensiones.end(), ext) == extensiones.end()) {
                            continue;
                        }

                        cout << "\nProcesando: " << rutaImg << endl;
                        map<string, float> datos = extraerValoresNutricionales(rutaImg);

                        if (datos.empty()) {
                            cout << "\t No se extrajo informacion, se omite.\n";
                            continue;
                        }

                        // Agregamos las caracteristicas extraidas de la imagen al csv
                        archivoCSV << datos["proteina"] << "," << datos["grasas_totales"] << "," << datos["grasas_trans"] << ","
                                   << datos["hidratos_disp"] << "," << datos["azucares"] << "," << datos["azucares_añadidos"] << ","
                                   << datos["fibra"] << "," << datos["sodio"] << "," << cat << "\n";

                        cout << "\t Aniadida al CSV.\n";
                    }
                }

                archivoCSV.close();
                cout << "\n\tExtraccion finalizada. Archivo guardado como '" << archivoSalida << "'.\n";
                break;
            }
            case 2: {
                string archivo;
                int nFolds = 7;
                cout << "\tIngrese el nombre del archivo CSV: ";
                cin >> archivo;

                Mat fullFeatureMat;
                Mat trainMat, trainLabelsMat, testMat, testLabelsMat;

                // Intentamos leer el archivo csv completo
                if (!syFeatureAndLabelMatrix_Read(archivo, fullFeatureMat)) {
                    cout << "Error al leer el archivo.\n";
                    continue;
                }

                // Separar datos (por default: 7 folds entrenamiento {nFolds}, 3 prueba)
                syFeatureAndLabelMatrix_Split(fullFeatureMat, trainMat, trainLabelsMat, testMat, testLabelsMat, nFolds);

                // Obtenemos el numero de clases automaticamente
                double minVal, maxVal;
                minMaxLoc(trainLabelsMat, &minVal, &maxVal);
                int nClasses = static_cast<int>(maxVal) + 1;

                // Mostramos los tamaños de las matrices
                cout << "\n--- TAMANIOS DE MATRICES ---\n";
                cout << "Entrenamiento - Caracteristicas: " << trainMat.rows << "x" << trainMat.cols << endl;
                cout << "Entrenamiento - Etiquetas:       " << trainLabelsMat.rows << "x" << trainLabelsMat.cols << endl;
                cout << "Prueba - Caracteristicas:        " << testMat.rows << "x" << testMat.cols << endl;
                cout << "Prueba - Etiquetas:              " << testLabelsMat.rows << "x" << testLabelsMat.cols << endl;

                // Mostramos la matriz de confusion
                Mat confusion(nClasses, nClasses, CV_32S, Scalar(0));

                // Entrenamiento y prueba de la ANN_MLP
                syANN_MLP_Train_and_Test(nClasses, trainMat, trainLabelsMat, testMat, testLabelsMat, confusion);

                // Guardamos el modelo en un .yml
                Ptr<ANN_MLP> modeloANN = ANN_MLP::create();

                // Asignamos el numero de capaz
                Mat_<int> layers(3,1);

                // Asignamos el numero de neuronas en cada capa
                layers(0) = trainMat.cols;
                layers(1) = trainMat.cols * 2 + 1;
                layers(2) = nClasses;

                // Asignamos los parametros anteiores a la ANN_MLP
                modeloANN->setLayerSizes(layers);
                modeloANN->setActivationFunction(ANN_MLP::SIGMOID_SYM);
                modeloANN->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
                modeloANN->setTrainMethod(ANN_MLP::BACKPROP, 0.0001);

                Mat train_classes = Mat::zeros(trainMat.rows, nClasses, CV_32FC1);
                for (int i = 0; i < train_classes.rows; i++)
                    train_classes.at<float>(i, trainLabelsMat.at<uchar>(i)) = 1.0;
                modeloANN->train(trainMat, ROW_SAMPLE, train_classes);
                modeloANN->save(modelo);
                cout << "Modelo guardado como '" << modelo << "'\n";
                break;
            }
            case 3: {
                string modelo = "modelo_final.yml";

                // Cargamos el lenguaje spanish para el OCR
                putenv((char*)"TESSDATA_PREFIX=C:\\Users\\carlo\\Desktop\\Inteligencia Artificial\\NutritionalClassifier\\tessdata");

                // Cargamos el modelo ya entrenado
                Ptr<ANN_MLP> ann = ANN_MLP::load(modelo);
                if (ann.empty()) {
                    cout << "Error: no se pudo cargar el modelo '" << modelo << "'\n";
                    continue;
                }

                // Solicitamos la imagen de prueba
                string rutaImagen;
                cout << "\n\tIngrese la ruta de la imagen nutrimental: ";
                cin >> rutaImagen;

                // Obtenemos los datos de la imagen (OCR)
                auto datos = extraerValoresNutricionales(rutaImagen);

                if (datos.empty()) {
                    cout << "\t\tNo se pudo obtener informacion valida de la imagen.\n";
                    continue;
                }

                // Guardamos los datos en una matriz
                Mat muestra = (Mat_<float>(1,8) << datos["proteina"], datos["grasas_totales"], datos["grasas_trans"],
                                                    datos["hidratos_disp"], datos["azucares"], datos["azucares_añadidos"],
                                                    datos["fibra"], datos["sodio"]);

                // Realizamos la clasificacion
                Mat salida;
                ann->predict(muestra, salida);
                Point clasePredicha;
                minMaxLoc(salida, 0, 0, 0, &clasePredicha);

                // Mostramos al usuario el resultado
                int categoria = clasePredicha.x;
                cout << "\n\tCategoria predicha: " << categoria << " -> ";

                switch (categoria) {
                    case 0: cout << "Consumo ocasional"; break;
                    case 1: cout << "Perdida de peso"; break;
                    case 2: cout << "Ganancia muscular"; break;
                    case 3: cout << "Energia/resistencia"; break;
                    case 4: cout << "Mantenimiento general"; break;
                    default: cout << "Desconocida"; break;
                }
                cout << endl;
                break;
            }
            case 4: {
                string modelo = "modelo_final.yml";

                // Intentamos cargar el modelo ya entrenado
                Ptr<ANN_MLP> ann = ANN_MLP::load(modelo);
                if (ann.empty()) {
                    cout << "Error: no se pudo cargar el modelo '" << modelo << "'\n";
                    continue;
                }

                string archivo;
                cout << "\n\tIngrese el nombre del archivo CSV con la muestra individual: ";
                cin >> archivo;

                ifstream input(archivo);
                if (!input.is_open()) {
                    cout << "Error: no se pudo abrir el archivo.\n";
                    continue;
                }

                // Leemos el archivo
                string linea;
                getline(input, linea);
                getline(input, linea); // Para leer la segunda linea (ignorar encabezados)
                input.close();

                // Obtenemos las caracteristicas separadas por comas (,)
                stringstream ss(linea);
                string valor;
                vector<float> datos;
                while (getline(ss, valor, ',')) {
                    datos.push_back(stof(valor));
                }

                if (datos.size() != 8) {
                    cout << "Error: se esperaban exactamente 8 valores en el CSV.\n";
                    continue;
                }

                // Guardamos las cracateristicas en una matriz
                Mat muestra = (Mat_<float>(1,8) << datos[0], datos[1], datos[2], datos[3],
                                                   datos[4], datos[5], datos[6], datos[7]);

                // Realizamos la prediccion
                Mat salida;
                ann->predict(muestra, salida);
                Point clasePredicha;
                minMaxLoc(salida, 0, 0, 0, &clasePredicha);

                // Mostramos al usuario el resultado
                int categoria = clasePredicha.x;
                cout << "\n\tCategoria predicha: " << categoria << " -> ";

                switch (categoria) {
                    case 0: cout << "Consumo ocasional"; break;
                    case 1: cout << "Perdida de peso"; break;
                    case 2: cout << "Ganancia muscular"; break;
                    case 3: cout << "Energia/resistencia"; break;
                    case 4: cout << "Mantenimiento general"; break;
                    default: cout << "Desconocida"; break;
                }
                cout << endl;
                break;
            }
            case 0:
                cout << "\nPrograma finalizado.\n";
                break;
            default:
                cout << "Opcion no valida.\n";
        }

    }while (opcion != 0);

    return 0;
}


// Funciones para la ANN_MLP
int syFeatureAndLabelMatrix_Read(string filename, Mat &fullFeatureMat){
   // WARNING: First line is assumed to contain feature labels!!!

   ifstream inputfile(filename);
   if(!inputfile) {cout << "\nsyFeatureAndLabelMatrix_Read(): Error reading input file."; return 0;}

   int nDataRead = 0;
   if (!fullFeatureMat.empty()) { cout << "\nReleasing current feature-and-label matrix."; fullFeatureMat.release(); }
   cout << "\n\nLoading feature-and-label matrix \""<<filename<<"\".\n";

   string current_line;
   getline(inputfile, current_line); // WARNING!!! We just discard first line

   // Array of float-type arrays
   vector< vector<float> > all_data;
   // Start reading lines as long as there are lines in the file
   while( getline(inputfile, current_line) )
   {
      // Now inside each line we need to seperate the cols
      vector<float> values;
      stringstream temp(current_line);
      string single_value;
      while(getline(temp,single_value,',')){
         values.push_back(atof(single_value.c_str())); // char* to float
      }
      // add the row to the complete data vector
      all_data.push_back(values);
   }
   inputfile.close();

   // Copy values into fullFeatureMat image
   fullFeatureMat = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_32FC1);
   for(int r = 0; r < fullFeatureMat.rows; r++)
      for(int c= 0; c < fullFeatureMat.cols; c++)
         fullFeatureMat.at<float>(r,c) = all_data[r][c];
   nDataRead = fullFeatureMat.rows*fullFeatureMat.cols;
   cout << "\nLoaded feature-and-label matrix with "<<fullFeatureMat.rows<<" samples ("<<nDataRead <<" data).\n";
   return nDataRead;
}

int syFeatureAndLabelMatrix_Split(const Mat &fullFeatureMat, Mat &trainMat, Mat &trainLabelsMat, Mat &testMat, Mat &testLabelsMat, int nFolds){
   // We get 1 of the n-folds of matrices        [ columns     x   rows          ]:
   //
   // trainMat:       A float-type image of size [ nFeatures   x   nTrainSamples ]
   // testMat:        A float-type image of size [ nFeatures   x   nTestSamples  ]
   // trainLabelsMat: A uchar-type image of size [     1       x   nTrainSamples ]
   // testLabelsMat:  A uchar-type image of size [     1       x   nTestSamples  ]
   //
   // from the matrix of features and labels,    [ nFeatures+1 x   nSamples      ]
   //

   if(fullFeatureMat.empty())
      { cout << "\nEmpty Feature-and-Label Matrix.\n"; return false; }
   if(!trainMat.empty()) trainMat.release();
   if(!trainLabelsMat.empty()) trainLabelsMat.release();
   if(!testMat.empty()) testMat.release();
   if(!testLabelsMat.empty()) testLabelsMat.release();

   Mat shuffled;
   syImShuffleRows(fullFeatureMat,shuffled);
//   syImWriteCSV_FEAT("shuffledMat.csv",shuffled);

   // Get 1 of n-folds
   int nSamplesPerFold = shuffled.rows/nFolds;
   int  nSamplesLearn = nSamplesPerFold*(nFolds-1);
   int nSamplesTest = shuffled.rows - nSamplesLearn;
   cout<<"\n\nFor "<<nFolds<<"-fold test: "<<nSamplesPerFold<< " samples per fold; "<<nSamplesLearn<<" for learning, "<<nSamplesTest<<" for testing.\n";

   trainMat = shuffled(Range(0,nSamplesLearn),Range(0,shuffled.cols-1));
   testMat  = shuffled(Range(nSamplesLearn,shuffled.rows),Range(0,shuffled.cols-1));
   trainLabelsMat  = shuffled(Range(0,nSamplesLearn),Range(shuffled.cols-1,shuffled.cols));
   testLabelsMat   = shuffled(Range(nSamplesLearn,shuffled.rows),Range(shuffled.cols-1,shuffled.cols));

   // From float to uchar data for labels
   trainLabelsMat.convertTo(trainLabelsMat, CV_8UC1);
   testLabelsMat.convertTo(testLabelsMat, CV_8UC1);
   return trainMat.cols;
}

int syImShuffleRows(const Mat &src, Mat &dst){
  std::vector <int> rowIndex;
  for (int r = 0; r < src.rows; r++)
    rowIndex.push_back(r);

  cv::randShuffle(rowIndex);

  for (int r = 0; r < src.rows; r++)
    dst.push_back(src.row(rowIndex[r]));

  return true;
}

int syANN_MLP_Train_and_Test(int nClasses, const Mat &trainMat, const Mat &trainLabelsMat, const Mat &testMat, const Mat &testLabelsMat, Mat &confusion){

   // CREATE ANN_MLP

   cout << "\nInitializing ANN_MLP\n\n";
   Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();

   Mat_<int> layers(3,1);
   int nFeatures = trainMat.cols;
   layers(0) = nFeatures;     // input
   layers(1) = nFeatures*2+1;  // hidden
   layers(2) = nClasses;  // output, 1 pin per class.

   ann->setLayerSizes(layers);
   ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM,0,0);
   ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));
   ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);


   // TRAIN THE NETWORK

   // Warning: ann requires "one-hot" encoding of class labels:
   // Class labels in a float-type sparse matrix of Samples x Classes
   // with an '1' in the corresponding correct-label column

   Mat train_classes = Mat::zeros(trainMat.rows, nClasses, CV_32FC1);
   for(int i=0; i<train_classes.rows; i++)
      train_classes.at<float>(i, trainLabelsMat.at<uchar>(i)) = 1.0;
   cout <<"\nTrain data size: "<< trainMat.size() << "\nTrain classes size: " << train_classes.size() << "\n\n";

   cout << "Training the ANN... (please wait)\n\n\n";
   ann->train(trainMat, ml::ROW_SAMPLE, train_classes);

////   // HERE, WE COUD'VE INCLUDED CODE FOR WRITNG THE MODEL...
////   ann->save("ANN_Model.yml");
////
////   //  AND, LATER READING THE MODEL
////   Ptr<ANN_MLP> annTRAINED = cv::ml::ANN_MLP::load("ANN_Model.yml");

   // TEST THE NETWORK

   cout << "ANN prediction test\n\n";

   // Test samples in testMat Mat
   for(int i=0; i<testMat.rows; i++)
   {
     int pred  = ann->predict(testMat.row(i), noArray());
     int truth = testLabelsMat.at<uchar>(i);
     confusion.at<int>(truth,pred) ++;
   }
   cout << "Confusion matrix:\n" << confusion << endl;

   Mat correct = confusion.diag();
   float accuracy = sum(correct)[0] / sum(confusion)[0];
   cout << "\nAccuracy: " << accuracy << "\n\n";
   return 1;
}


// Funciones para el OCR
//      Preprocesar imagen y extraer texto usando Tesseract OCR
map<string, float> extraerValoresNutricionales(const string& rutaImagen) {
    // Obtenemos la imagen y la guardanmos en una matriz
    Mat imagen = imread(rutaImagen);
    if (imagen.empty()) {
        cerr << "Error: no se pudo cargar la imagen.\n";
        return {};
    }

    // Convertimos a escala de grises y aplicamos el CLAHE
    Mat gris;
    cvtColor(imagen, gris, COLOR_BGR2GRAY);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gris, gris);

    // Binarizamos la imagen
    Mat binaria;
    threshold(gris, binaria, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    // Cambiamos dimensiones para un mejor OCR
    resize(binaria, binaria, Size(), 2.5, 2.5, INTER_LINEAR);

    // Inicializacion del OCR
    tesseract::TessBaseAPI ocr;
    if (ocr.Init(NULL, "spa", tesseract::OEM_LSTM_ONLY)) {
        cerr << "Error al inicializar Tesseract.\n";
        return {};
    }
    ocr.SetPageSegMode(tesseract::PSM_AUTO);
    ocr.SetVariable("user_defined_dpi", "300");
    ocr.SetVariable("tessedit_char_whitelist", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áéíóúÁÉÍÓÚñÑ .,:gmgkcal");

    // Aplicamos el OCR
    ocr.SetImage(binaria.data, binaria.cols, binaria.rows, 1, binaria.step);
    string textoOCR = ocr.GetUTF8Text();

    // Limpiamos el resultado
    string texto = limpiarTextoOCR(textoOCR);
    cout << "\n\t- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << endl;
    cout << "\n\tTexto OCR detectado:\n\n" << texto << endl;
    cout << "\t- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << endl;

    // Limpiamos texto
    texto = regex_replace(texto, regex("[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\\.,\\s]+"), " ");
    transform(texto.begin(), texto.end(), texto.begin(), ::tolower);

    // Diccionario de nutrientes
    map<string, float> nutrientes = {
        {"proteina", 0}, {"grasas_totales", 0}, {"grasas_trans", 0},
        {"hidratos_disp", 0}, {"azucares", 0}, {"azucares_añadidos", 0},
        {"fibra", 0}, {"sodio", 0}
    };

    // Expresion regular para capturar datos
    smatch match;
    regex patron("([a-záéíóúñ\\s]+):?\\s*([\\d]+[\\.,]?\\d*)\\s*(g|mg)?");

    // Obtenemos posibles coincidencias
    istringstream stream(texto);
    string linea;
    while (getline(stream, linea)) {
        if (regex_search(linea, match, patron)) {
            string campo = match[1];
            string valor = regex_replace(string(match[2]), regex(","), ".");
            float numero = stof(valor);
            if (match[3] == "mg") numero /= 1000.0f;

            // Asignar campo
            if (campo.find("prote") != string::npos) nutrientes["proteina"] = numero;
            else if ((campo.find("total") != string::npos || campo.find("totales") != string::npos) && campo.find("grasa") != string::npos) nutrientes["grasas_totales"] = numero;
            else if (campo.find("saturada") != string::npos) nutrientes["grasas_totales"] += numero;
            else if (campo.find("trans") != string::npos) nutrientes["grasas_trans"] = numero;
            else if (campo.find("hidrato") != string::npos || campo.find("carbo") != string::npos) nutrientes["hidratos_disp"] = numero;
            else if (campo.find("añadi") != string::npos || campo.find("anad") != string::npos) nutrientes["azucares_añadidos"] = numero;
            else if (campo.find("azuc") != string::npos) nutrientes["azucares"] = numero;
            else if (campo.find("fibra") != string::npos) nutrientes["fibra"] = numero;
            else if (campo.find("sodio") != string::npos) nutrientes["sodio"] = numero;
        }
    }

    // Mostramos al usuario el resultado
    cout << "\n\tValores extraidos:\n";
    for (const auto& [clave, valor] : nutrientes) {
        cout << "\t\t" << clave << ": " << valor << " g\n";
    }

    return nutrientes;
}

// Limpiar el texto OCR de caracteres raros
string limpiarTextoOCR(const string& texto) {
    string limpio = texto;

    // Eliminamos caracteres no imprimibles
    limpio = regex_replace(limpio, regex("[^\\x20-\\x7EáéíóúÁÉÍÓÚñÑ\\s\\.,:gmgkcal]+"), " ");

    // Reemplazamos guiones mal interpretados
    limpio = regex_replace(limpio, regex("[-=]{2,}"), " ");

    // Eliminamos multiples espacios
    limpio = regex_replace(limpio, regex(" {2,}"), " ");

    // Convertimos todo a minusculas
    transform(limpio.begin(), limpio.end(), limpio.begin(), ::tolower);

    return limpio;
}
