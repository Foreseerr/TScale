#include "stdafx.h"
#include "data_config.h"
#include <gpt/data/text_saveload.h>
#include <gpt/data/net_data.h>


void TDataSourceConfigParser::ParseScript(const TVector<TConfigFile::TOp> &opArr)
{
    for (yint ptr = 0; ptr < YSize(opArr); ++ptr) {
        const TConfigFile::TOp &op = opArr[ptr];
        if (op.Op == CFG_OP_ASSIGNMENT) {
            if (op.Dst == "TEST_FRACTION") {
                TestFraction = atof(op.Args[0].c_str());
            } else if (op.Dst == "USE_PPM") {
                UsePPM = (IsYes(op.Args[0]));
            } else if (op.Dst == "USE_LMATCH") {
                UseLMatch = (IsYes(op.Args[0]));
            } else {
                DebugPrintf("unknown variable %s\n", op.Dst.c_str());
                abort();
            }

        } else if (op.Op == CFG_OP_CALL) {
            // tokenizer ops
            if (op.Dst == "set_vocab_size") {
                Y_VERIFY(YSize(op.Args) >= 1);
                VocabSize = atoi(op.Args[0].c_str());
                if (YSize(op.Args) > 1) {
                    FragmentStartToken = VocabSize++; // we expect multiple system tokens, add one for now
                }

            } else if (op.Dst == "set_doc_start_token") {
                Y_VERIFY(YSize(op.Args) == 1);
                DocStartToken = atoi(op.Args[0].c_str());

            } else if (op.Dst == "load_tokenizer") {
                Y_VERIFY(YSize(op.Args) == 1);
                Serialize(IO_READ, op.Args[0], Tokenizer);
                LoadTokenizerParams();

            } else if (op.Dst == "make_byte_tokenizer") {
                Tokenizer.MakeByteEncoder(TTokenizer::TK_CHAR);
                LoadTokenizerParams();

                // dataset ops
            } else if (op.Dst == "make_char_dataset") {
                Y_VERIFY(Tokenizer.IsEmpty());
                Y_VERIFY(DataBuild.Get() == 0);
                TVector<char> text;
                LoadDocument(&text, op.Args[0]);
                Dataset = MakeCharDataset(&Tokenizer, text, TestFraction, UsePPM, UseLMatch);
                LoadTokenizerParams();

            } else if (op.Dst == "connect_data_server") {
                Y_VERIFY(YSize(op.Args) == 1);
                Dataset = ConnectDataServer(NNet::CreateTcpSendRecv(), op.Args[0]);

            } else if (op.Dst == "connect_http_data_server") {
                Y_VERIFY(YSize(op.Args) == 1);
                Dataset = ConnectHttpDataServer(op.Args[0]);

            } else if (op.Dst == "load_tokenized_train" || op.Dst == "load_tokenized_test") {
                yint tokenWidth = 2;
                if (YSize(op.Args) > 1) {
                    tokenWidth = atoi(op.Args[1].c_str());
                }
                TVector<TBPEToken> data;
                LoadTokenized(op.Args[0], tokenWidth, 0, &data);
                CreateDatasetBuilder();
                float ltTestFraction = (op.Dst == "load_tokenized_train") ? 0 : 1;
                TDatasetParams params(VocabSize);
                params.CountDocset(data, 0, YSize(data), ltTestFraction);
                float weight = 1;
                DataBuild->AddTokenizedDocset(data, params, weight);

            } else if (op.Dst == "load_text" || op.Dst == "load_folder" || op.Dst == "load_docset") {
                Y_VERIFY(!Tokenizer.IsEmpty());
                Y_VERIFY(YSize(op.Args) > 0);
                TVector<TVector<char>> docSet;
                if (op.Dst == "load_text") {
                    docSet.resize(1);
                    LoadDocument(&docSet[0], op.Args[0]);
                } else if (op.Dst == "load_folder") {
                    LoadDocumentSetFromFiles(&docSet, op.Args[0]);
                } else if (op.Dst == "load_docset") {
                    LoadDocumentSetFromBin(&docSet, op.Args[0]);
                }
                CreateDatasetBuilder();
                float weight = (YSize(op.Args) > 1) ? atof(op.Args[1].c_str()) : 1;
                AddDocset(DataBuild.Get(), Tokenizer, docSet, weight, TestFraction);

            } else if (op.Dst == "load_indexed_docset_folder") {
                Y_VERIFY(YSize(op.Args) > 0);
                CreateDatasetBuilder();
                float weight = (YSize(op.Args) > 1) ? atof(op.Args[1].c_str()) : 1;
                AddIndexedDocset(DataBuild.Get(), op.Args[0], weight);

            } else if (op.Dst == "index_docset_folder") {
                Y_VERIFY(YSize(op.Args) == 1);
                Y_VERIFY(!Tokenizer.IsEmpty());
                IndexDocsetDir(op.Args[0], Tokenizer, UsePPM, TestFraction);

            } else if (op.Dst == "index_tokenized_folder") {
                Y_VERIFY(YSize(op.Args) == 3);
                Y_VERIFY(Tokenizer.IsEmpty() && "tokenizer is not expected, using already tokenized dataset");
                yint tokenWidth = atoi(op.Args[1].c_str());
                yint headerSize = atoi(op.Args[2].c_str());
                IndexTokenizedDir(op.Args[0], VocabSize, DocStartToken, UsePPM, TestFraction, tokenWidth, headerSize);

            } else if (op.Dst == "set_lmatch_index_folder") {
                LMatchIndexDir = op.Args[0];

            } else {
                DebugPrintf("unknown function %s\n", op.Dst.c_str());
                abort();
            }
        }
    }
    MakeDataset();
}
