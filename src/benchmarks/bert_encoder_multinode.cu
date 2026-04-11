/**
 * bert_encoder_multinode.cu
 *
 * Multi-node BERT encoder layer. MPI distributes attention heads across nodes,
 * each node uses CtPipeline for intra-node GPU parallelism.
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
#include "../multi_gpu/pipeline/ct_pipeline.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;
using namespace nexus_multi_gpu;

struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() { return chrono::duration<double,milli>(chrono::high_resolution_clock::now()-t0).count(); }
};

static void level_refresh(CKKSEvaluator &e, PhantomCiphertext &ct, double s) {
    PhantomPlaintext pt; vector<double> v;
    e.decryptor.decrypt(ct,pt); e.encoder.decode(pt,v);
    e.encoder.encode(v,s,pt); e.encryptor.encrypt(pt,ct);
}

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

    int gpn=4, heads=16, inner=16, seq=16;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--gpus-per-node")&&i+1<argc) gpn=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--heads")&&i+1<argc) heads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--inner")&&i+1<argc) inner=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seq-len")&&i+1<argc) seq=atoi(argv[++i]);
    }
    int total_gpus=world*gpn, per_rank=heads/world, hidden=heads*32;

    if(rank==0){
        printf("════════════════════════════════════════════════════════════\n");
        printf("  BERT Encoder Layer — Multi-Node (%d nodes, %d GPUs)\n",world,total_gpus);
        printf("  heads=%d (%d/node), hidden=%d, inner=%d\n",heads,per_rank,hidden,inner);
        printf("════════════════════════════════════════════════════════════\n\n");
    }

    PerfTimer timer;
    size_t N=1ULL<<16;
    vector<int> cb; cb.push_back(58); for(int i=0;i<20;i++) cb.push_back(40); cb.push_back(58);
    double SCALE=(double)(1ULL<<40);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N,cb));

    cudaSetDevice(0);
    PhantomContext ctx(parms);

    // Broadcast SK
    PhantomSecretKey sk0(ctx); stringstream skb;
    if(rank==0) sk0.save(skb);
    {string d; if(rank==0) d=skb.str(); int sz=d.size();
     MPI_Bcast(&sz,1,MPI_INT,0,MPI_COMM_WORLD);
     if(rank!=0) d.resize(sz);
     MPI_Bcast(&d[0],sz,MPI_CHAR,0,MPI_COMM_WORLD);
     if(rank!=0){skb.str(d);skb.clear();}}
    PhantomSecretKey sk; skb.seekg(0); sk.load(skb);

    PhantomCKKSEncoder enc(ctx);
    PhantomPublicKey pk=sk.gen_publickey(ctx);
    PhantomRelinKey rk=sk.gen_relinkey(ctx);
    PhantomGaloisKey gk=sk.create_galois_keys(ctx);
    size_t slots=enc.slot_count();

    CKKSEvaluator eval0(&ctx,&pk,&sk,&enc,&rk,&gk,SCALE);

    // Weights (same seed on all ranks)
    mt19937 rng(42); uniform_real_distribution<double> wd(-0.02,0.02),id(-0.5,0.5);
    auto mkw=[&](){vector<vector<double>> w(inner,vector<double>(slots,0.0));
        for(auto&r:w) for(size_t s=0;s<(size_t)hidden;s++) r[s]=wd(rng); return w;};
    auto Wq=mkw(),Wk=mkw(),Wv=mkw(),Wo=mkw(),Wf1=mkw(),Wf2=mkw();

    // Pipeline
    CtPipeline pipe=CtPipeline::create(parms,gpn,sk);
    pipe.enable_galois_keys();

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) printf("[Setup] Done on all nodes\n");

    // Scatter encrypted heads
    timer.start();
    vector<PhantomCiphertext> local_cts;
    if(rank==0){
        vector<PhantomCiphertext> all(heads);
        for(int i=0;i<heads;i++){
            vector<double> d(slots,0.0);
            for(size_t s=0;s<(size_t)hidden;s++) d[s]=id(rng);
            PhantomPlaintext pt; eval0.encoder.encode(d,SCALE,pt);
            eval0.encryptor.encrypt(pt,all[i]);
        }
        for(int i=0;i<per_rank;i++) local_cts.push_back(all[i]);
        for(int r=1;r<world;r++){
            vector<PhantomCiphertext> b(all.begin()+r*per_rank,all.begin()+(r+1)*per_rank);
            string d=ser_cts(b); int sz=d.size();
            MPI_Send(&sz,1,MPI_INT,r,0,MPI_COMM_WORLD);
            MPI_Send(d.data(),sz,MPI_CHAR,r,1,MPI_COMM_WORLD);
        }
    } else {
        int sz; MPI_Recv(&sz,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        string d(sz,'\0'); MPI_Recv(&d[0],sz,MPI_CHAR,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        local_cts=deser_cts(d);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_ms=timer.elapsed_ms();
    if(rank==0) printf("[1] Scatter: %.1f ms (%d heads/node)\n",scatter_ms,per_rank);

    // Execute full BERT layer per-head on each node
    timer.start();
    pipe.scatter(local_cts);
    pipe.execute_full([&](int gpu,PhantomContext &c,PhantomSecretKey &lsk,
                          PhantomPublicKey &lpk,PhantomRelinKey &lrk,
                          PhantomGaloisKey &lgk,PhantomCKKSEncoder &e,
                          vector<PhantomCiphertext> &local){
        CKKSEvaluator le(&c,&lpk,&lsk,&e,&lrk,&lgk,SCALE);
        GELUEvaluator lg(le); SoftmaxEvaluator ls(le); LNEvaluator ll(le); MMEvaluator lm(le);

        for(auto &ct:local){
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
            level_refresh(le,po[0],SCALE);

            PhantomCiphertext ln1o; ll.layer_norm(po[0],ln1o,hidden);
            level_refresh(le,ln1o,SCALE);

            vector<PhantomCiphertext> fi={ln1o},fo;
            lm.matrix_mul_unified(fi,Wf1,1,fo);
            PhantomCiphertext go; lg.gelu(fo[0],go);
            if(go.coeff_modulus_size()<=2) level_refresh(le,go,SCALE);

            vector<PhantomCiphertext> f2i={go},f2o;
            lm.matrix_mul_unified(f2i,Wf2,1,f2o);
            level_refresh(le,f2o[0],SCALE);

            PhantomCiphertext ln2o; ll.layer_norm(f2o[0],ln2o,hidden);
            level_refresh(le,ln2o,SCALE);
            ct=std::move(ln2o);
        }
    });
    cudaSetDevice(0);
    auto results=pipe.gather();
    MPI_Barrier(MPI_COMM_WORLD);
    double compute_ms=timer.elapsed_ms();
    if(rank==0) printf("[2] Compute: %.1f ms\n",compute_ms);

    // Gather
    timer.start();
    if(rank==0){
        for(int r=1;r<world;r++){
            int sz; MPI_Recv(&sz,1,MPI_INT,r,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            string d(sz,'\0'); MPI_Recv(&d[0],sz,MPI_CHAR,r,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    } else {
        string d=ser_cts(results); int sz=d.size();
        MPI_Send(&sz,1,MPI_INT,0,0,MPI_COMM_WORLD);
        MPI_Send(d.data(),sz,MPI_CHAR,0,1,MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double gather_ms=timer.elapsed_ms();
    if(rank==0){
        printf("[3] Gather: %.1f ms\n",gather_ms);
        printf("\n════════════════════════════════════════════════\n");
        printf("  Multi-Node BERT Encoder Layer\n");
        printf("════════════════════════════════════════════════\n");
        printf("  Nodes: %d, GPUs: %d, Heads: %d\n",world,total_gpus,heads);
        printf("  Scatter: %.1f ms\n",scatter_ms);
        printf("  Compute: %.1f ms\n",compute_ms);
        printf("  Gather:  %.1f ms\n",gather_ms);
        printf("  Total:   %.1f ms\n",scatter_ms+compute_ms+gather_ms);
        printf("════════════════════════════════════════════════\n");
    }

    pipe.destroy();
    MPI_Finalize();
    return 0;
}

#else
#include <cstdio>
int main(){printf("ERROR: No MPI\n");return 1;}
#endif
