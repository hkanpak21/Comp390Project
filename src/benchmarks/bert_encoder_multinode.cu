/**
 * bert_encoder_multinode.cu
 *
 * Multi-node BERT encoder layer with REAL bootstrapping.
 * MPI distributes attention heads across nodes, each node uses
 * per-GPU threading for intra-node parallelism.
 *
 * Usage:
 *   srun --nodes=4 --ntasks-per-node=1 --gres=gpu:4 \
 *     ./bin/bert_encoder_multinode --gpus-per-node 4 --heads 16 --inner 16
 */

#ifdef USE_MPI

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <thread>
#include <atomic>
#include <algorithm>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "ckks_evaluator.cuh"
#include "gelu.cuh"
#include "softmax.cuh"
#include "layer_norm.cuh"
#include "matrix_mul.cuh"
#include "bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() { return chrono::duration<double,milli>(chrono::high_resolution_clock::now()-t0).count(); }
};

static string ser_cts(const vector<PhantomCiphertext> &c) {
    stringstream s; int n=c.size(); s.write((char*)&n,4);
    for(auto&ct:c) ct.save(s); return s.str();
}
static vector<PhantomCiphertext> deser_cts(const string &d) {
    stringstream s(d); int n; s.read((char*)&n,4);
    vector<PhantomCiphertext> c(n); for(auto&ct:c) ct.load(s); return c;
}

int main(int argc, char **argv) {
    MPI_Init(&argc,&argv);
    int rank,world; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&world);

    int gpn=4, heads=16, inner=16, seq=16, hidden=64;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--gpus-per-node")&&i+1<argc) gpn=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--heads")&&i+1<argc) heads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--inner")&&i+1<argc) inner=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seq-len")&&i+1<argc) seq=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--hidden")&&i+1<argc) hidden=atoi(argv[++i]);
    }
    int total_gpus=world*gpn, per_rank=heads/world;

    if(rank==0){
        printf("════════════════════════════════════════════════════════════\n");
        printf("  BERT Encoder Layer — Multi-Node Real Bootstrap\n");
        printf("  %d nodes × %d GPUs = %d GPUs, heads=%d (%d/node)\n",world,gpn,total_gpus,heads,per_rank);
        printf("════════════════════════════════════════════════════════════\n\n");
    }

    PerfTimer timer;

    // ═══ Parameters ═══
    long logN=15, logn=logN-2, logNh=logN-1;
    size_t N=1ULL<<logN;
    long sparse_slots_val=1L<<logn;
    int logp=46, logq=51, log_special=51;
    int main_mod=21, bs_mod=14;
    int total_level=main_mod+bs_mod;
    double SCALE=pow(2.0,logp);

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for(int i=0;i<main_mod;i++) coeff_bits.push_back(logp);
    for(int i=0;i<bs_mod;i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N,coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    long boundary_K=25, deg=59, scale_factor=2, inverse_deg=1, loge=10;

    // Setup on GPU 0 of each node
    cudaSetDevice(0);
    PhantomContext ctx0(parms);
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey sk_local(ctx0);
    size_t slots=enc0.slot_count();

    // Broadcast SK from rank 0
    stringstream skb;
    if(rank==0) sk_local.save(skb);
    {string d; if(rank==0) d=skb.str(); int sz=d.size();
     MPI_Bcast(&sz,1,MPI_INT,0,MPI_COMM_WORLD);
     if(rank!=0) d.resize(sz);
     MPI_Bcast(&d[0],sz,MPI_CHAR,0,MPI_COMM_WORLD);
     if(rank!=0){skb.str(d);skb.clear();}}
    PhantomSecretKey sk; skb.seekg(0); sk.load(skb);

    PhantomPublicKey pk=sk.gen_publickey(ctx0);
    PhantomRelinKey rk=sk.gen_relinkey(ctx0);
    PhantomGaloisKey gk0;
    CKKSEvaluator eval0(&ctx0,&pk,&sk,&enc0,&rk,&gk0,SCALE);

    // Serialize SK for GPU distribution
    stringstream sk_ser;
    sk.save(sk_ser);
    string sk_str = sk_ser.str();

    // Weights (same seed on all ranks)
    mt19937 rng(42); uniform_real_distribution<double> wd(-0.02,0.02),id(-0.5,0.5);
    auto mkw=[&](){vector<vector<double>> w(inner,vector<double>(slots,0.0));
        for(auto&r:w) for(size_t s=0;s<std::min((size_t)hidden,slots);s++) r[s]=wd(rng); return w;};
    auto Wq=mkw(),Wk=mkw(),Wv=mkw(),Wo=mkw(),Wf1=mkw(),Wf2=mkw();

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) printf("[Setup] Done on all nodes\n");

    // ═══ Scatter encrypted heads ═══
    timer.start();
    vector<string> local_ct_data;
    if(rank==0){
        // Encrypt all heads
        vector<PhantomCiphertext> all(heads);
        for(int i=0;i<heads;i++){
            vector<double> d(slots,0.0);
            for(size_t s=0;s<std::min((size_t)hidden,slots);s++) d[s]=id(rng);
            PhantomPlaintext pt; eval0.encoder.encode(d,SCALE,pt);
            eval0.encryptor.encrypt(pt,all[i]);
            for(int j=0;j<bs_mod;j++) eval0.evaluator.mod_switch_to_next_inplace(all[i]);
        }
        // Keep rank 0's share
        for(int i=0;i<per_rank;i++){
            stringstream ss; all[i].save(ss);
            local_ct_data.push_back(ss.str());
        }
        // Send to other ranks
        for(int r=1;r<world;r++){
            vector<PhantomCiphertext> b(all.begin()+r*per_rank,all.begin()+(r+1)*per_rank);
            string d=ser_cts(b); int sz=d.size();
            MPI_Send(&sz,1,MPI_INT,r,0,MPI_COMM_WORLD);
            MPI_Send(d.data(),sz,MPI_CHAR,r,1,MPI_COMM_WORLD);
        }
    } else {
        int sz; MPI_Recv(&sz,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        string d(sz,'\0'); MPI_Recv(&d[0],sz,MPI_CHAR,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        auto cts=deser_cts(d);
        for(auto &ct:cts){
            stringstream ss; ct.save(ss);
            local_ct_data.push_back(ss.str());
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_ms=timer.elapsed_ms();
    if(rank==0) printf("[1] Scatter: %.1f ms (%d heads/node)\n",scatter_ms,per_rank);

    // ═══ Per-node multi-GPU compute ═══
    timer.start();

    // Assign heads to GPUs within this node
    int local_heads = (int)local_ct_data.size();
    vector<vector<int>> gpu_heads(gpn);
    for(int i=0;i<local_heads;i++) gpu_heads[i%gpn].push_back(i);

    vector<vector<string>> gpu_results(gpn);
    vector<thread> threads;
    atomic<int> setup_done{0};
    PerfTimer compute_timer;

    for(int g=0;g<gpn;g++){
        threads.emplace_back([&,g](){
            cudaSetDevice(g);
            PhantomContext ctx(parms);
            PhantomCKKSEncoder enc(ctx);

            PhantomSecretKey lsk;
            { stringstream ss(sk_str); lsk.load(ss); }
            PhantomPublicKey lpk=lsk.gen_publickey(ctx);
            PhantomRelinKey lrk=lsk.gen_relinkey(ctx);

            vector<int> gsteps;
            gsteps.push_back(0);
            for(int i=0;i<logN-1;i++) gsteps.push_back(1<<i);
            for(int i=0;i<logN-1;i++) gsteps.push_back(-(1<<i));
            gsteps.push_back(-seq);
            gsteps.push_back(-hidden);

            PhantomGaloisKey lgk=lsk.create_galois_keys_from_steps(ctx,gsteps);
            CKKSEvaluator le(&ctx,&lpk,&lsk,&enc,&lrk,&lgk,SCALE);

            Bootstrapper lb(loge,logn,logNh,total_level,SCALE,boundary_K,deg,scale_factor,inverse_deg,&le);
            lb.slot_vec.push_back(logn);
            lb.prepare_mod_polynomial();
            lb.generate_LT_coefficient_3();

            gsteps.clear();
            gsteps.push_back(0);
            for(int i=0;i<logN-1;i++) gsteps.push_back(1<<i);
            for(int i=0;i<logN-1;i++) gsteps.push_back(-(1<<i));
            gsteps.push_back(-seq);
            gsteps.push_back(-hidden);
            lb.addLeftRotKeys_Linear_to_vector_3(gsteps);
            le.decryptor.create_galois_keys_from_steps(gsteps,*le.galois_keys);

            int my_count=setup_done.fetch_add(1)+1;
            while(setup_done.load()<gpn){/* spin */}
            if(my_count==gpn) compute_timer.start();

            GELUEvaluator lg(le); SoftmaxEvaluator ls(le); LNEvaluator ll(le); MMEvaluator lm(le);

            for(int h_idx:gpu_heads[g]){
                PhantomCiphertext ct;
                { stringstream ss(local_ct_data[h_idx]); ct.load(ss); }

                vector<PhantomCiphertext> xi={ct},q,k,v;
                lm.matrix_mul_unified(xi,Wq,1,q);
                lm.matrix_mul_unified(xi,Wk,1,k);
                lm.matrix_mul_unified(xi,Wv,1,v);

                le.evaluator.mod_switch_to_inplace(k[0],q[0].chain_index());
                k[0].set_scale(q[0].scale());
                PhantomCiphertext as;
                le.evaluator.multiply(q[0],k[0],as);
                le.evaluator.relinearize_inplace(as,*le.relin_keys);
                le.evaluator.rescale_to_next_inplace(as);

                PhantomCiphertext aw; ls.softmax(as,aw,seq);

                le.evaluator.mod_switch_to_inplace(v[0],aw.chain_index());
                v[0].set_scale(aw.scale());
                PhantomCiphertext ao;
                le.evaluator.multiply(aw,v[0],ao);
                le.evaluator.relinearize_inplace(ao,*le.relin_keys);
                le.evaluator.rescale_to_next_inplace(ao);

                vector<PhantomCiphertext> pi={ao},po;
                lm.matrix_mul_unified(pi,Wo,1,po);

                while(po[0].coeff_modulus_size()>1) le.evaluator.mod_switch_to_next_inplace(po[0]);
                PhantomCiphertext b1; lb.bootstrap_3(b1,po[0]);

                PhantomCiphertext ln1o; ll.layer_norm(b1,ln1o,hidden);
                while(ln1o.coeff_modulus_size()>1) le.evaluator.mod_switch_to_next_inplace(ln1o);
                PhantomCiphertext b2; lb.bootstrap_3(b2,ln1o);

                vector<PhantomCiphertext> fi={b2},fo;
                lm.matrix_mul_unified(fi,Wf1,1,fo);
                PhantomCiphertext go; lg.gelu(fo[0],go);

                vector<PhantomCiphertext> f2i={go},f2o;
                lm.matrix_mul_unified(f2i,Wf2,1,f2o);

                while(f2o[0].coeff_modulus_size()>1) le.evaluator.mod_switch_to_next_inplace(f2o[0]);
                PhantomCiphertext b3; lb.bootstrap_3(b3,f2o[0]);

                PhantomCiphertext ln2o; ll.layer_norm(b3,ln2o,hidden);
                while(ln2o.coeff_modulus_size()>1) le.evaluator.mod_switch_to_next_inplace(ln2o);
                PhantomCiphertext b4; lb.bootstrap_3(b4,ln2o);

                stringstream ss; b4.save(ss);
                gpu_results[g].push_back(ss.str());
            }
            cudaDeviceSynchronize();
        });
    }
    for(auto &t:threads) t.join();
    double compute_ms=compute_timer.elapsed_ms();
    MPI_Barrier(MPI_COMM_WORLD);
    double node_ms=timer.elapsed_ms();

    // ═══ Gather results ═══
    timer.start();
    if(rank==0){
        for(int r=1;r<world;r++){
            int sz; MPI_Recv(&sz,1,MPI_INT,r,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            string d(sz,'\0'); MPI_Recv(&d[0],sz,MPI_CHAR,r,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    } else {
        // Collect all GPU results for this rank
        vector<PhantomCiphertext> results;
        for(int g=0;g<gpn;g++){
            for(auto &s:gpu_results[g]){
                PhantomCiphertext ct; stringstream ss(s); ct.load(ss);
                results.push_back(std::move(ct));
            }
        }
        cudaSetDevice(0);
        string d=ser_cts(results); int sz=d.size();
        MPI_Send(&sz,1,MPI_INT,0,0,MPI_COMM_WORLD);
        MPI_Send(d.data(),sz,MPI_CHAR,0,1,MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double gather_ms=timer.elapsed_ms();

    if(rank==0){
        printf("\n════════════════════════════════════════════════\n");
        printf("  Multi-Node BERT Encoder — Real Bootstrap\n");
        printf("════════════════════════════════════════════════\n");
        printf("  Nodes: %d, GPUs: %d, Heads: %d\n",world,total_gpus,heads);
        printf("  Scatter:   %8.1f ms\n",scatter_ms);
        printf("  Compute:   %8.1f ms (per-node, %d heads)\n",compute_ms,per_rank);
        printf("  Node total:%8.1f ms (includes setup)\n",node_ms);
        printf("  Gather:    %8.1f ms\n",gather_ms);
        printf("  Total:     %8.1f ms\n",scatter_ms+node_ms+gather_ms);
        printf("════════════════════════════════════════════════\n");
    }

    MPI_Finalize();
    return 0;
}

#else
#include <cstdio>
int main(){printf("ERROR: No MPI\n");return 1;}
#endif
