#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <sw/redis++/redis++.h>
#include <faiss/IndexFlat.h> 
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using namespace sw::redis;

class RAG {
public:
    RAG(const std::string& redis_host = "127.0.0.1", int redis_port = 6379);
    ~RAG() = default;
    
    std::string generate_response(const std::string& query, 
                                const std::vector<std::string>& document_keys);
    std::vector<std::string> get_document_keys();
    
private:
    Redis redis;
    py::object torch;
    py::object sentence_transformers;
    py::object model;
    py::object llm;
    py::object tokenizer;
    bool initialized = false;
    
    void initialize_models();
    std::vector<std::pair<std::string, std::vector<float>>> get_docs_and_embeddings(
        const std::vector<std::string>& keys);
    std::vector<std::string> retrieve_relevant_docs(
        const std::string& query,
        const std::vector<std::string>& document_keys);
    std::vector<float> get_query_embedding(const std::string& query);
    std::string generate_answer(const std::string& query, const std::string& context);
};

void RAG::initialize_models() {
    if (!initialized) {
        try {
            torch = py::module::import("torch");
            sentence_transformers = py::module::import("sentence_transformers");
            model = sentence_transformers.attr("SentenceTransformer")("all-mpnet-base-v2");

            auto transformers = py::module::import("transformers");
            std::string token = std::getenv("HF_TOKEN") ? std::getenv("HF_TOKEN") : "token";
            llm = transformers.attr("AutoModelForCausalLM").attr("from_pretrained")("meta-llama/Llama-2-7b-hf", py::arg("token") = token);
            tokenizer = transformers.attr("AutoTokenizer").attr("from_pretrained")("meta-llama/Llama-2-7b-hf", py::arg("token") = token);
            
            tokenizer.attr("pad_token") = tokenizer.attr("eos_token");
            tokenizer.attr("pad_token_id") = tokenizer.attr("eos_token_id");
            initialized = true;
            std::cout << "Modelos estáticos inicializados" << std::endl;
        } catch (const py::error_already_set& e) {
            throw std::runtime_error("Error inicializando los módulos Python: " + std::string(e.what()));
        }
    }
}

RAG::RAG(const std::string& redis_host, int redis_port) 
    : redis("tcp://" + redis_host + ":" + std::to_string(redis_port)) {
    try {
        auto pong = redis.ping();
        std::cout << "Conexión a Redis exitosa: " << pong << std::endl;
        initialize_models();
    } catch (const Error& e) {
        throw std::runtime_error("Error conectando a Redis: " + std::string(e.what()));
    }
}

std::vector<float> RAG::get_query_embedding(const std::string& query) {
    auto embeddings = model.attr("encode")(std::vector<std::string>{query}, 
                                          py::arg("convert_to_tensor") = true);
    auto numpy_array = embeddings.attr("cpu")().attr("numpy")();
    auto array = numpy_array.cast<py::array_t<float>>();
    
    auto buffer = array.request();
    float* ptr = static_cast<float*>(buffer.ptr);
    size_t embedding_size = buffer.shape[1];
    
    std::vector<float> embedding(embedding_size);
    for (size_t j = 0; j < embedding_size; j++) {
        embedding[j] = ptr[j];
    }
    
    return embedding;
}

std::vector<std::pair<std::string, std::vector<float>>> RAG::get_docs_and_embeddings(
    const std::vector<std::string>& keys) {
    std::vector<std::pair<std::string, std::vector<float>>> docs_and_embs;
    
    for (const auto& key : keys) {
        OptionalString emb_str = redis.get(key + ":emb");
        OptionalString text = redis.get(key + ":text");
        
        if (!emb_str || !text) {
            std::cout << "Advertencia: No se encontraron embedding o texto para la clave " << key << std::endl;
            continue;
        }
        
        std::vector<float> embedding;
        std::stringstream ss_emb(*emb_str);
        std::string value;
        while (std::getline(ss_emb, value, ',')) {
            embedding.push_back(std::stof(value));
        }
        
        docs_and_embs.push_back({*text, embedding});
    }
    
    if (docs_and_embs.empty()) {
        std::cout << "Error: No se encontraron documentos válidos en Redis" << std::endl;
    }
    return docs_and_embs;
}

std::vector<std::string> RAG::retrieve_relevant_docs(
    const std::string& query,
    const std::vector<std::string>& document_keys) {
    auto query_embedding = get_query_embedding(query);
    auto docs_and_embs = get_docs_and_embeddings(document_keys);
    
    if (docs_and_embs.empty()) {
        std::cout << "Error: No hay documentos para procesar" << std::endl;
        return {};
    }
    
    int d = query_embedding.size();
    faiss::IndexFlatIP index(d);
    std::vector<float> all_embeddings;
    for (const auto& doc : docs_and_embs) {
        all_embeddings.insert(all_embeddings.end(), doc.second.begin(), doc.second.end());
    }
    index.add(docs_and_embs.size(), all_embeddings.data());
    
    float distances[1];
    faiss::idx_t indices[1];
    index.search(1, query_embedding.data(), 1, distances, indices);
    
    std::cout << "Documento seleccionado: " << document_keys[indices[0]] << " con similitud: " << distances[0] << std::endl;
    return {docs_and_embs[indices[0]].first};
}

std::string RAG::generate_answer(const std::string& query, const std::string& context) {
    //std::cout << "Iniciando generación de respuesta..." << std::endl;
    std::string prompt = "Usando el contexto, responde la pregunta en una sola oración clara y completa:\nPregunta: " + query + "\nContexto: " + context + "\nRespuesta: ";
    //std::cout << "Prompt creado: " << prompt << std::endl;
    
    auto inputs = tokenizer.attr("encode_plus")(prompt, 
                                                py::arg("return_tensors") = "pt", 
                                                py::arg("padding") = true, 
                                                py::arg("truncation") = true, 
                                                py::arg("max_length") = 150);
    auto input_ids = inputs["input_ids"];
    auto attention_mask = inputs["attention_mask"];
    //std::cout << "Input IDs y attention mask generados" << std::endl;
    
    auto outputs = llm.attr("generate")(input_ids, 
                                        py::arg("attention_mask") = attention_mask,
                                        py::arg("max_new_tokens") = 40,
                                        py::arg("num_return_sequences") = 1,
                                        py::arg("no_repeat_ngram_size") = 4,
                                        py::arg("temperature") = 0.7,
                                        py::arg("top_k") = 40,
                                        py::arg("top_p") = 0.9,
                                        py::arg("do_sample") = true,
                                        py::arg("eos_token_id") = tokenizer.attr("eos_token_id").cast<int>());
    //std::cout << "Output generado" << std::endl;
    
    std::string output_size = outputs.attr("size")().attr("__str__")().cast<std::string>();
    //std::cout << "Forma de outputs: " << output_size << std::endl;
    
    auto adjusted_outputs = outputs.attr("squeeze")(0).attr("to")("cpu");
    //std::cout << "Outputs ajustado en Python" << std::endl;
    
    std::string adjusted_size = adjusted_outputs.attr("size")().attr("__str__")().cast<std::string>();
    //std::cout << "Forma de outputs ajustado: " << adjusted_size << std::endl;
    
    auto response = tokenizer.attr("decode")(adjusted_outputs, py::arg("skip_special_tokens") = true);
    std::string final_response = response.cast<std::string>();
    size_t pos = final_response.find("Respuesta: ");
    if (pos != std::string::npos) {
        final_response = final_response.substr(pos + 11);
    }
    pos = final_response.find(".");
    if (pos != std::string::npos) {
        final_response = final_response.substr(0, pos + 1);
    }
    //std::cout << "Respuesta decodificada: " << final_response << std::endl;
    
    return final_response;
}

std::string RAG::generate_response(const std::string& query, 
                                  const std::vector<std::string>& document_keys) {
    py::gil_scoped_acquire acquire;
    std::cout << "Recuperando documentos relevantes..." << std::endl;
    auto relevant_docs = retrieve_relevant_docs(query, document_keys);
    std::cout << "Documentos recuperados, generando respuesta..." << std::endl;
    if (relevant_docs.empty()) {
        return "No se encontraron documentos relevantes.";
    }
    return generate_answer(query, relevant_docs[0]);
}

// Nuevo método para obtener todas las claves de documentos
std::vector<std::string> RAG::get_document_keys() {
    std::set<std::string> document_keys_set; // Usar set para evitar duplicados
    std::vector<std::string> keys_buffer;
    
    // Usar SCAN para iterar sobre las claves
    auto cursor = redis.scan(0, "doc:*", 100, std::back_inserter(keys_buffer));
    
    while (true) {
        for (const auto& key : keys_buffer) {
            // Extraer la parte base (ej. "doc:1" de "doc:1:text" o "doc:1:emb")
            std::string base_key = key.substr(0, key.find_last_of(":"));
            document_keys_set.insert(base_key);
        }
        
        if (cursor == 0) { // Fin de la iteración
            break;
        }
        
        keys_buffer.clear();
        cursor = redis.scan(cursor, "doc:*", 100, std::back_inserter(keys_buffer));
    }
    
    // Convertir set a vector
    std::vector<std::string> document_keys(document_keys_set.begin(), document_keys_set.end());
    std::sort(document_keys.begin(), document_keys.end()); // Ordenar opcional
    
    std::cout << "Se encontraron " << document_keys.size() << " claves de documentos." << std::endl;
    return document_keys;
}

PYBIND11_MODULE(rag_module, m) {
    py::class_<RAG>(m, "RAG")
        .def(py::init<const std::string&, int>(), 
             py::arg("redis_host") = "127.0.0.1", 
             py::arg("redis_port") = 6379)
        .def("generate_response", &RAG::generate_response)
        .def("get_document_keys", &RAG::get_document_keys); 
}
