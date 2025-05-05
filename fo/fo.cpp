#include <string>
#include <vector>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <algorithm>
#include <unordered_map>

// default - program, z.cfg 
// LIBRARY()  (default is program)
// NOPCH(filename)
// DEP(some/library,another/library)

using namespace std;

typedef string TString;
typedef long long yint;
typedef unsigned char ui8;
typedef unsigned short ui16;
typedef unsigned int ui32;
typedef unsigned long long ui64;
typedef ofstream TOFStream;

#define TVector vector
#define THashMap unordered_map

#define Y_ASSERT assert

template <class T>
yint YSize(const T &col)
{
    return (yint)col.size();
}

template <class T, class TElem>
inline bool IsInSet(const T &c, const TElem &e) { return find(c.begin(), c.end(), e) != c.end(); }

inline void Out(TOFStream &f, const TString &s) { f << s.c_str(); }

//////////////////////////////////////////////////////////////////////////////////////////////
struct TFindFileResult
{
    TString Name;
    bool IsDir = false;

    TFindFileResult() {}
    TFindFileResult(const TString &n, bool isDir) : Name(n), IsDir(isDir) {}
};

#ifdef _MSC_VER
#include <windows.h> // for findfirst

static void FindAllFiles(const TString &prefix, TVector<TFindFileResult> *res)
{
    res->resize(0);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((prefix + "/*.*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    if (fd.cFileName[0] != '.') {
        res->push_back(TFindFileResult(fd.cFileName, (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)));
    }
    while (FindNextFileA(h, &fd)) {
        if (fd.cFileName[0] != '.') {
            res->push_back(TFindFileResult(fd.cFileName, (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)));
        }
    }
    FindClose(h);
}

void MakeDirectory(const TString &dir)
{
    CreateDirectoryA(dir.c_str(), 0);
}

#else
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
//#include <unistd.h>
//#include <libgen.h>

static void FindAllFiles(const TString &prefix, TVector<TFindFileResult> *res)
{
    res->resize(0);
    DIR *dir = opendir(prefix.c_str());
    if (dir == NULL) {
        Y_ASSERT(0); // directory does not exist?
        return;
    }
    for (;;) {
        struct dirent *dp = readdir(dir);
        if (dp == 0) {
            break;
        }
        if (dp->d_name[0] == '.') {
            continue;
        }
        struct stat fprop;
        TString fname(dp->d_name);
        int rv = stat((prefix + "/" + fname).c_str(), &fprop);
        if (S_ISDIR(fprop.st_mode)) {
            res->push_back(TFindFileResult(fname, true));
        } else {
            res->push_back(TFindFileResult(fname, false));
        }
    }
    closedir(dir);
}

void MakeDirectory(const TString &dir)
{
    mkdir(dir.c_str(), 0777);
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////
enum EFolderType
{
    FT_PROGRAM,
    FT_LIBRARY,
};

enum ESourceType
{
    ST_H,
    ST_CPP,
    ST_CUH,
    ST_CU,
};

enum EPchType
{
    PCH_GEN,
    PCH_USE,
    PCH_NONE
};

struct TSourceFile
{
    TString Name;
    ESourceType Type = ST_CPP;
    EPchType Pch = PCH_USE;
};

struct TFolderPath
{
    TVector<TString> PathArr;
};

struct TConfigCommand
{
    TString Cmd;
    TVector<TString> Args;
};

struct TSourceFolder
{
    TString Folder;
    EFolderType FType = FT_PROGRAM;
    TFolderPath Path;
    bool IsUsingCuda = false;
    TVector<TSourceFile> Files;
    TVector<TFolderPath> DepArr;
    TVector<TConfigCommand> Config;
};


//////////////////////////////////////////////////////////////////////////////////////////////
TString Sprintf(const char *pszFormat, ...)
{
    TString res;

    va_list va;
    va_start(va, pszFormat);
    yint len = vsnprintf(0, 0, pszFormat, va);
    res.resize(len + 1);
    yint resLen = vsnprintf(&res[0], YSize(res), pszFormat, va);
    res.resize(resLen);
    va_end(va);
    //
    return res;
}

static TString MakeString(char c)
{
    TString res;
    res.push_back(c);
    return res;
}


static bool EndsWith(TString &a, const TString &b)
{
    yint bsz = YSize(b);
    yint idx = YSize(a) - bsz;
    if (idx < 0) {
        return false;
    }
    for (yint i = 0; i < bsz; ++i) {
        if (a[idx + i] != b[i]) {
            return false;
        }
    }
    return true;
}

static TString GetStringPath(const TFolderPath &fp)
{
    if (fp.PathArr.empty()) {
        return "";
    }
    TString res = fp.PathArr[0];
    for (yint i = 1; i < YSize(fp.PathArr); ++i) {
        res += "/" + fp.PathArr[i];
    }
    return res;
}

static TFolderPath GetFolderPath(const TString &fp)
{
    TFolderPath res;
    TString folder;
    for (char c : fp) {
        if (c == '/') {
            res.PathArr.push_back(folder);
            folder = "";
        } else {
            folder += c;
        }
    }
    res.PathArr.push_back(folder);
    return res;
}

inline bool operator==(const TFolderPath &a, const TFolderPath &b)
{
    return a.PathArr == b.PathArr;
}

inline bool operator!=(const TFolderPath &a, const TFolderPath &b)
{
    return a.PathArr != b.PathArr;
}

static TString GetProjectName(const TFolderPath &fp)
{
    if (fp.PathArr.empty()) {
        return "root"; // never happens?
    }
    TString res = fp.PathArr[0];
    for (yint i = 1; i < YSize(fp.PathArr); ++i) {
        res += "-" + fp.PathArr[i];
    }
    return res;
}

//////////////////////////////////////////////////////////////////////////////////////////////
static void ParseZCfg(const TString &prefix, TVector<TConfigCommand> *res)
{
    res->resize(0);
    ifstream f(prefix + "/z.cfg");
    if (!f.good()) {
        printf("failed to read z.cfg at %s\n", prefix.c_str());
        abort();
    }
    enum ETokenState {
        SPACE,
        TOKEN,
    };
    enum ECmdState {
        CMD,
        CMD_OPEN,
        ARGS,
    };

    TVector<TString> tokenArr;
    for (yint nLine = 1; f.good(); ++nLine) {
        TString token;
        const yint LINE_SIZE = 10000;
        static char szLine[LINE_SIZE];
        f.getline(szLine, LINE_SIZE);
        ETokenState state = SPACE;
        for (char *p = szLine; *p; ++p) {
            char c = *p;
            switch (state) {
            case SPACE:
                if (isalnum(c)) {
                    state = TOKEN;
                    token.push_back(c);
                } else if (!isspace(c)) {
                    tokenArr.push_back(MakeString(c));
                }
                break;
            case TOKEN:
                if (isspace(c)) {
                    tokenArr.push_back(token);
                    token = "";
                    state = SPACE;
                } else if (c == '(' || c == ')') {
                    tokenArr.push_back(token);
                    tokenArr.push_back(MakeString(c));
                    token = "";
                    state = SPACE;
                } else {
                    token += c;
                }
                break;
            }
        }
        if (state == TOKEN) {
            tokenArr.push_back(token);
        }
    }

    TConfigCommand cc;
    ECmdState ps = CMD;
    for (const TString &token : tokenArr) {
        switch (ps) {
        case CMD:
            if (token == "(" || token == ")") {
                printf("%s/z.cfg no command before ()\n", prefix.c_str());
                abort();
            }
            cc.Cmd = token;
            ps = CMD_OPEN;
            break;
        case CMD_OPEN:
            if (token != "(") {
                printf("%s/z.cfg expected ( after %s\n", prefix.c_str(), cc.Cmd.c_str());
                abort();
            }
            ps = ARGS;
            break;
        case ARGS:
            if (token == ")") {
                res->push_back(cc);
                cc = TConfigCommand();
                ps = CMD;
            } else if (token == "(") {
                printf("%s/z.cfg expected command %s arguments\n", prefix.c_str(), cc.Cmd.c_str());
                abort();
            } else {
                cc.Args.push_back(token);
            }
            break;
        }
    }
}


void ParseSourceDir(const TString &prefix, TSourceFolder *p, TVector<TSourceFolder> *resFolders);

static void AddFile(const TString &fnameArg, const TString &prefix, TSourceFolder *res, TVector<TSourceFolder> *resFolders)
{
    TString fname = fnameArg;
    for (char &c : fname) {
        c = (char)tolower((ui8)c);
    }
    TSourceFile sf;
    sf.Name = fnameArg;
    if (fname == "z.cfg") {
        ParseZCfg(prefix, &res->Config);
    } else if (EndsWith(fname, ".cpp")) {
        sf.Type = ST_CPP;
        res->Files.push_back(sf);
    } else if (EndsWith(fname, ".h")) {
        sf.Type = ST_H;
        res->Files.push_back(sf);
    } else if (EndsWith(fname, ".cu")) {
        sf.Type = ST_CU;
        res->Files.push_back(sf);
        res->IsUsingCuda = true;
    } else if (EndsWith(fname, ".cuh")) {
        sf.Type = ST_CUH;
        res->Files.push_back(sf);
        res->IsUsingCuda = true;
    } else {
        printf("Unkown file %s at %s\n", fnameArg.c_str(), prefix.c_str());
        abort();
    }
}

static void AddDir(const TString &dirname, const TString &prefix, TSourceFolder *res, TVector<TSourceFolder> *resFolders)
{
    TSourceFolder xx;
    xx.Folder = dirname;
    xx.Path = res->Path;
    xx.Path.PathArr.push_back(xx.Folder);
    ParseSourceDir(prefix + "/" + dirname, &xx, resFolders);
    if (!xx.Files.empty()) {
        TFolderPath util = GetFolderPath("util");
        if (xx.Path != util) {
            xx.DepArr.push_back(util);
        }
        resFolders->push_back(xx);
    }
}

void ParseSourceDir(const TString &prefix, TSourceFolder *res, TVector<TSourceFolder> *resFolders)
{
    TVector<TSourceFolder> newFolders;
    TVector<TFindFileResult> allFiles;
    FindAllFiles(prefix, &allFiles);
    for (const TFindFileResult &ff : allFiles) {
        if (ff.IsDir) {
            AddDir(ff.Name, prefix, res, &newFolders);
        } else {
            AddFile(ff.Name, prefix, res, &newFolders);
        }
    }
    // checks
    if (!res->Files.empty() && !newFolders.empty()) {
        printf("folder %s should have either files or subfolders\n", GetStringPath(res->Path).c_str());
        abort();
    }
    if (res->Files.empty() && !res->Config.empty()) {
        printf("empty folder %s should have no config file\n", GetStringPath(res->Path).c_str());
        abort();
    }
    resFolders->insert(resFolders->end(), newFolders.begin(), newFolders.end());
}


//////////////////////////////////////////////////////////////////////////////////////////////
static void ApplyConfig(TVector<TSourceFolder> *pProjArr)
{
    THashMap<TString, yint> projId;
    for (yint id = 0; id < YSize(*pProjArr); ++id) {
        TSourceFolder &proj = (*pProjArr)[id];
        projId[GetStringPath(proj.Path)] = id;
    }
    for (TSourceFolder &proj : *pProjArr) {
        for (const TConfigCommand &cc : proj.Config) {
            if (cc.Cmd == "LIBRARY") {
                proj.FType = FT_LIBRARY;
            } else if (cc.Cmd == "NOPCH") {
                for (const TString &fname : cc.Args) {
                    bool found = false;
                    for (yint i = 0, sz = YSize(proj.Files); i < sz; ++i) {
                        if (proj.Files[i].Name == fname) {
                            proj.Files[i].Pch = PCH_NONE;
                            found = true;
                        }
                    }
                    if (!found) {
                        printf("file %s from NOPCH at %s does not exist\n", fname.c_str(), GetStringPath(proj.Path).c_str());
                        abort();
                    }
                }
            } else if (cc.Cmd == "DEP") {
                for (const TString &dep : cc.Args) {
                    TFolderPath fp = GetFolderPath(dep);
                    if (projId.find(GetStringPath(fp)) == projId.end()) {
                        printf("DEP at %s refers non existing library %s\n", GetStringPath(proj.Path).c_str(), dep.c_str());
                        abort();
                    }
                    proj.DepArr.push_back(fp);
                }
            } else {
                printf("%s config unknown command %s\n", GetStringPath(proj.Path).c_str(), cc.Cmd.c_str());
                abort();
            }
        }
        for (TSourceFile &sf : proj.Files) {
            if (sf.Name == "stdafx.cpp") {
                sf.Pch = PCH_GEN;
            }
        }
    }
    // check that dependencies only on libs
    for (TSourceFolder &proj : *pProjArr) {
        for (const TFolderPath &fp : proj.DepArr) {
            yint id = projId[GetStringPath(fp)];
            const TSourceFolder &depProj = (*pProjArr)[id];
            if (depProj.FType != FT_LIBRARY) {
                printf("project %s depends on non-library %s\n", GetStringPath(proj.Path).c_str(), GetStringPath(depProj.Path).c_str());
                abort();
            }
        }
    }
}


static void PropagateUsingCuda(TVector<TSourceFolder> *pProjArr)
{
    THashMap<TString, yint> projId;
    for (yint id = 0; id < YSize(*pProjArr); ++id) {
        TSourceFolder &proj = (*pProjArr)[id];
        projId[GetStringPath(proj.Path)] = id;
    }
    for (;;) {
        bool hasFinished = true;
        for (TSourceFolder &proj : *pProjArr) {
            if (proj.IsUsingCuda) {
                continue;
            }
            for (const TFolderPath &fp : proj.DepArr) {
                auto it = projId.find(GetStringPath(fp));
                if (it == projId.end()) {
                    abort(); // deparr references non existing project
                } else if ((*pProjArr)[it->second].IsUsingCuda) {
                    hasFinished = false;
                    proj.IsUsingCuda = true;
                    break;
                }
            }
        }
        if (hasFinished) {
            break;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
struct TBuildOrder
{
    TVector<yint> Order;
    TVector<yint> OrderPlace;

    TBuildOrder(yint projCount)
    {
        OrderPlace.resize(projCount, -1);
    }
    void AddToTail(yint id)
    {
        yint place = OrderPlace[id];
        if (place >= 0) {
            Order[place] = -1;
        }
        OrderPlace[id] = YSize(Order);
        Order.push_back(id);
    }
};

static void MakeProgDependOnLibs(TVector<TSourceFolder> *pProjArr)
{
    THashMap<TString, yint> projId;
    yint projCount = YSize(*pProjArr);
    for (yint id = 0; id < projCount; ++id) {
        TSourceFolder &proj = (*pProjArr)[id];
        projId[GetStringPath(proj.Path)] = id;
    }
    for (TSourceFolder &proj : *pProjArr) {
        if (proj.FType == FT_LIBRARY) {
            continue;
        }
        TVector<yint> depDepth;
        depDepth.resize(projCount, 0);
        TBuildOrder order(projCount);
        for (const TFolderPath &fp : proj.DepArr) {
            yint depId = projId[GetStringPath(fp)];
            depDepth[depId] = 1;
            order.AddToTail(depId);
        }
        bool isUsingCuda = proj.IsUsingCuda;
        for (yint iter = 0;; ++iter) {
            bool hasChanged = false;
            for (yint k = 0; k < YSize(order.Order); ++k) {
                yint id = order.Order[k];
                if (id >= 0) {
                    yint depth = depDepth[id];
                    const TSourceFolder &depProj = (*pProjArr)[id];
                    isUsingCuda |= depProj.IsUsingCuda;
                    for (const TFolderPath &fp : depProj.DepArr) {
                        yint depId = projId[GetStringPath(fp)];
                        if (depDepth[depId] < depth + 1) {
                            depDepth[depId] = depth + 1;
                            order.AddToTail(depId);
                            hasChanged = true;
                        }
                    }
                }
            }
            if (!hasChanged) {
                break;
            }
            if (iter > projCount) {
                printf("project %s has circular dependencies for ?\n", GetProjectName(proj.Path).c_str());
                abort();
            }
        }
        proj.DepArr.resize(0);
        for (yint depId : order.Order) {
            if (depId >= 0) {
                proj.DepArr.push_back((*pProjArr)[depId].Path);
            }
        }
        proj.IsUsingCuda = isUsingCuda;
    }
    for (TSourceFolder &proj : *pProjArr) {
        if (proj.FType == FT_LIBRARY) {
            proj.DepArr.resize(0);
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
struct TMSGuid
{
    union {
        ui8 Data[16];
        ui16 Data16[8];
        ui64 Data64[2];
    };
    TMSGuid() { memset(this, 0, sizeof(*this)); }
};

template<class TRng>
static void MakeRandomGuid(TRng &rng, TMSGuid *res)
{
    for (yint i = 0; i < 16; ++i) {
        res->Data[i] = rng.Uniform(255); // not 256 on purpose
    }
}

struct TSomeHash
{
    ui64 Arr[8];

    TSomeHash()
    {
        Arr[0] = 0x3b832379d2035f59ull;
        Arr[1] = 0x6bfbddf0b28a6967ull;
        Arr[2] = 0xf6949075f9061d59ull;
        Arr[3] = 0x507d44b9b65649a1ull;
        Arr[4] = 0xa0d5977ba7a270adull;
        Arr[5] = 0x0d05fa68d1c2fa3full;
        Arr[6] = 0x1b9d078df6504af1ull;
        Arr[7] = 0x08be71a21634ce73ull;
    }
    TMSGuid Hash(const void *data, yint len)
    {
        TMSGuid res;
        const char *cData = (const char *)data;
        ui64 add = 13;
        for (yint blkId = 0; blkId < 8; ++blkId) {
            ui64 hh = 0;
            for (yint i = 0; i < len; ++i) {
                hh = (hh + cData[i]) * Arr[blkId] + add;
            }
            res.Data16[blkId] = hh >> (64 - 16);
            add = hh;
        }
        return res;
    }
};

static TMSGuid GetPathGuid(const TFolderPath &path)
{
    TString spath = GetStringPath(path);
    TSomeHash sh;
    return sh.Hash(&spath[0], YSize(spath));
}

static TMSGuid Xor(const TMSGuid &a, const TMSGuid &b)
{
    TMSGuid res;
    for (yint i = 0; i < 16; ++i) {
        res.Data[i] = a.Data[i] ^ b.Data[i];
    }
    return res;
}

static TString Format(const TMSGuid &g)
{
    return Sprintf("{%04x%04x-%04x-%04x-%04x-%04x%04x%04x}", g.Data16[0], g.Data16[1], g.Data16[2], g.Data16[3], g.Data16[4], g.Data16[5], g.Data16[6], g.Data16[7]);
}



//////////////////////////////////////////////////////////////////////////////////////////////
struct TXmlProp
{
    TString Name, Value;

    TXmlProp() {}
    TXmlProp(const TString &name, const TString &val) : Name(name), Value(val) {}
};

struct TXmlNode
{
    TString Name;
    TVector<TXmlProp> Props;
    TString MyText;
    TVector<TXmlNode> Children;

    TXmlNode() {}
    TXmlNode(const TString &n) : Name(n) {}
    TXmlNode& Prop(const TString &name, const TString &val)
    {
        Props.push_back(TXmlProp(name, val));
        return *this;
    }
    TXmlNode &Text(const TString &tx)
    {
        MyText = tx;
        return *this;
    }
    void Add(const TXmlNode &xml) { Children.push_back(xml); }
};


static void RenderXml(TOFStream &f, int indent, const TXmlNode &node)
{
    for (yint i = 0; i < indent; ++i) {
        f << " ";
    }
    f << "<" << node.Name.c_str();
    for (const TXmlProp &prop : node.Props) {
        f << " " << prop.Name << "=\"" << prop.Value << "\"";
    }
    if (node.Children.empty() && node.MyText.empty()) {
        f << " />\n";
        return;
    }
    f << ">";
    if (node.Children.empty()) {
        f << node.MyText.c_str();
        f << "</" << node.Name.c_str() << ">\n";
        return;
    }
    if (!node.MyText.empty()) {
        printf("xml error\n");
        abort();
    }
    f << "\n";
    for (const TXmlNode &nn : node.Children) {
        RenderXml(f, indent + 2, nn);
    }
    for (yint i = 0; i < indent; ++i) {
        f << " ";
    }
    f << "</" << node.Name.c_str() << ">\n";
}

static void RenderXml(const TString &dst, const TXmlNode &root)
{
    TOFStream f(dst.c_str());
    f << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
    RenderXml(f, 0, root);
}


//////////////////////////////////////////////////////////////////////////////////////////////
static void PrintSln(const TString &dst, const TVector<TSourceFolder> &projArr)
{
    TOFStream f(dst.c_str());
    f << (char)0xef << (char)0xbb << (char)0xbf << "\n";
    f << "Microsoft Visual Studio Solution File, Format Version 12.00\n";
    f << "# Visual Studio Version 17\n";
    f << "VisualStudioVersion = 17.4.33110.190\n";
    f << "MinimumVisualStudioVersion = 10.0.40219.1\n";

    TMSGuid xz;
    xz.Data64[0] = 0xdeadbeaf;
    xz.Data64[1] = 0x31331;

    for (const TSourceFolder &proj : projArr) {
        TString xzGuid = Format(Xor(xz, GetPathGuid(proj.Path)));
        TString projGuid = Format(GetPathGuid(proj.Path));
        Out(f, Sprintf("Project(\"%s\") = \"%s\", \"%s/%s.vcxproj\", \"%s\"\n",
            xzGuid.c_str(),
            GetProjectName(proj.Path).c_str(),
            GetStringPath(proj.Path).c_str(),
            GetProjectName(proj.Path).c_str(),
            projGuid.c_str()
        ));
        //if (!proj.DepArr.empty()) {
        //    f << "\tProjectSection(ProjectDependencies) = postProject\n";
        //    for (const TFolderPath &dep : proj.DepArr) {
        //        TString projGuid = Format(GetPathGuid(dep));
        //        Out(f, Sprintf("\t\t%s = %s\n", projGuid.c_str(), projGuid.c_str()));
        //    }
        //    f << "\tEndProjectSection\n";
        //}
        f << "EndProject\n";
    }
    f << "Global\n";
    f << "\tGlobalSection(SolutionConfigurationPlatforms) = preSolution\n";
    f << "\t\tDebug|x64 = Debug|x64\n";
    f << "\t\tRelease|x64 = Release|x64\n";
    f << "\tEndGlobalSection\n";
    f << "\tGlobalSection(ProjectConfigurationPlatforms) = postSolution\n";
    for (const TSourceFolder &proj : projArr) {
        TString g = Format(GetPathGuid(proj.Path));
        Out(f, Sprintf("\t\t%s.Debug|x64.ActveCfg=Debug|x64\n", g.c_str()));
        Out(f, Sprintf("\t\t%s.Debug|x64.Build=Debug|x64\n", g.c_str()));
        Out(f, Sprintf("\t\t%s.Release|x64.ActveCfg=Release|x64\n", g.c_str()));
        Out(f, Sprintf("\t\t%s.Release|x64.Build=Release|x64\n", g.c_str()));
    }
    f << "\tEndGlobalSection\n";
    f << "\tGlobalSection(SolutionProperties) = preSolution\n";
    f << "\t\tHideSolutionNode = FALSE\n";
    f << "\tEndGlobalSection\n";
    f << "\tGlobalSection(ExtensibilityGlobals) = postSolution\n";
    f << "\t\tSolutionGuid = {8E4E74CF-90D0-4B2F-A470-96FDCEAE40EB}\n";
    f << "\tEndGlobalSection\n";
    f << "EndGlobal\n";
}


static TXmlNode MakeConfiguration(const TString &mode, const TString &arch)
{
    TXmlNode res = TXmlNode("ProjectConfiguration").Prop("Include", Sprintf("%s|%s", mode.c_str(), arch.c_str()));
    res.Add(TXmlNode("Configuration").Text(mode));
    res.Add(TXmlNode("Platform").Text(arch));
    return res;
}

static void PrintVcproj(const TSourceFolder &proj, const TString &vcProjFile, const TString &pathToSrc, const TString &utilPath)
{
    TXmlNode root("Project");
    root.Prop("DefaultTargets", "Build");
    root.Prop("xmlns", "http://schemas.microsoft.com/developer/msbuild/2003");

    TXmlNode pconf = TXmlNode("ItemGroup").Prop("Label", "ProjectConfigurations");
    pconf.Add(MakeConfiguration("Debug", "x64"));
    pconf.Add(MakeConfiguration("Release", "x64"));
    root.Add(pconf);

    TXmlNode globals = TXmlNode("PropertyGroup").Prop("Label", "Globals");
    globals.Add(TXmlNode("VCProjectVersion").Text("16.0"));
    globals.Add(TXmlNode("Keyword").Text("Win32Proj"));
    globals.Add(TXmlNode("ProjectGuid").Text(Format(GetPathGuid(proj.Path))));
    globals.Add(TXmlNode("RootNamespace").Text(GetProjectName(proj.Path)));
    globals.Add(TXmlNode("WindowsTargetPlatformVersion").Text("10.0"));
    globals.Add(TXmlNode("ProjectName").Text(GetProjectName(proj.Path)));
    root.Add(globals);
    
    root.Add(TXmlNode("Import").Prop("Project", "$(VCTargetsPath)\\Microsoft.Cpp.Default.props"));

    TXmlNode cfgType;
    if (proj.FType == FT_LIBRARY) {
        cfgType = TXmlNode("ConfigurationType").Text("StaticLibrary");
    } else {
        cfgType = TXmlNode("ConfigurationType").Text("Application");
    }

    TXmlNode cfgDebug = TXmlNode("PropertyGroup").Prop("Condition", "'$(Configuration)|$(Platform)'=='Debug|x64'").Prop("Label", "Configuration");
    cfgDebug.Add(cfgType);
    cfgDebug.Add(TXmlNode("UseDebugLibraries").Text("true"));
    cfgDebug.Add(TXmlNode("PlatformToolset").Text("v143"));
    cfgDebug.Add(TXmlNode("CharacterSet").Text("Unicode"));
    root.Add(cfgDebug);

    TXmlNode cfgRelease = TXmlNode("PropertyGroup").Prop("Condition", "'$(Configuration)|$(Platform)'=='Release|x64'").Prop("Label", "Configuration");
    cfgRelease.Add(cfgType);
    cfgRelease.Add(TXmlNode("UseDebugLibraries").Text("false"));
    cfgRelease.Add(TXmlNode("PlatformToolset").Text("v143"));
    cfgRelease.Add(TXmlNode("WholeProgramOptimization").Text("true"));
    cfgRelease.Add(TXmlNode("CharacterSet").Text("Unicode"));
    root.Add(cfgRelease);

    root.Add(TXmlNode("Import").Prop("Project", "$(VCTargetsPath)\\Microsoft.Cpp.props"));

    if (proj.IsUsingCuda) {
        TXmlNode ext = TXmlNode("ImportGroup").Prop("Label", "ExtensionSettings");
        ext.Add(TXmlNode("Import").Prop("Project", "$(VCTargetsPath)\\BuildCustomizations\\CUDA 12.6.props"));
        root.Add(ext);
    }

    TXmlNode include("PropertyGroup");
    include.Add(TXmlNode("IncludePath").Text("$(VC_IncludePath);$(WindowsSDK_IncludePath);" + utilPath));
    root.Add(include);

    if (proj.IsUsingCuda) {
        TXmlNode disableIncrementalLink = TXmlNode("PropertyGroup").Prop("Condition", "'$(Configuration)|$(Platform)'=='Debug|x64'");
        disableIncrementalLink.Add(TXmlNode("LinkIncremental").Text("false"));
        root.Add(disableIncrementalLink);
    }

    TString linkWhole;
    for (const TFolderPath &dep : proj.DepArr) {
        linkWhole += " /WHOLEARCHIVE:" + GetProjectName(dep);
    }

    TXmlNode debugProp = TXmlNode("ItemDefinitionGroup").Prop("Condition", "'$(Configuration)|$(Platform)'=='Debug|x64'");
    TXmlNode debugPropCompile("ClCompile");
    debugPropCompile.Add(TXmlNode("WarningLevel").Text("Level3"));
    debugPropCompile.Add(TXmlNode("SDLCheck").Text("true"));
    debugPropCompile.Add(TXmlNode("PreprocessorDefinitions").Text("_DEBUG;%(PreprocessorDefinitions)")); // _LIB; ??
    debugPropCompile.Add(TXmlNode("ConformanceMode").Text("true"));
    debugPropCompile.Add(TXmlNode("PrecompiledHeader").Text("Use"));
    debugPropCompile.Add(TXmlNode("PrecompiledHeaderFile").Text("stdafx.h"));
    debugPropCompile.Add(TXmlNode("SupportJustMyCode").Text("false"));
    debugPropCompile.Add(TXmlNode("DebugInformationFormat").Text("ProgramDatabase"));
    debugProp.Add(debugPropCompile);
    TXmlNode debugPropLink("Link");
    debugPropLink.Add(TXmlNode("GenerateDebugInformation").Text("true"));
    debugPropLink.Add(TXmlNode("AdditionalOptions").Text(linkWhole));
    debugProp.Add(debugPropLink);
    if (proj.IsUsingCuda) {
        TXmlNode debugCuda("CudaCompile");
        debugCuda.Add(TXmlNode("TargetMachinePlatform").Text("64"));
        debugCuda.Add(TXmlNode("CodeGeneration").Text("compute_89,sm_89"));
        debugCuda.Add(TXmlNode("FastMath").Text("true"));
        debugProp.Add(debugCuda);
    }
    root.Add(debugProp);

    TXmlNode releaseProp = TXmlNode("ItemDefinitionGroup").Prop("Condition", "'$(Configuration)|$(Platform)'=='Release|x64'");
    TXmlNode releasePropCompile("ClCompile");
    releasePropCompile.Add(TXmlNode("WarningLevel").Text("Level3"));
    releasePropCompile.Add(TXmlNode("FunctionLevelLinking").Text("true"));
    releasePropCompile.Add(TXmlNode("IntrinsicFunctions").Text("true"));
    releasePropCompile.Add(TXmlNode("SDLCheck").Text("true"));
    releasePropCompile.Add(TXmlNode("PreprocessorDefinitions").Text("NDEBUG;%(PreprocessorDefinitions)")); // _LIB; ??
    releasePropCompile.Add(TXmlNode("ConformanceMode").Text("true"));
    releasePropCompile.Add(TXmlNode("PrecompiledHeader").Text("Use"));
    releasePropCompile.Add(TXmlNode("PrecompiledHeaderFile").Text("stdafx.h"));
    //releasePropCompile.Add(TXmlNode("DebugInformationFormat").Text("OldStyle"));
    releasePropCompile.Add(TXmlNode("Optimization").Text("Full"));
    releasePropCompile.Add(TXmlNode("InlineFunctionExpansion").Text("AnySuitable"));
    releasePropCompile.Add(TXmlNode("EnableEnhancedInstructionSet").Text("AdvancedVectorExtensions"));
    releaseProp.Add(releasePropCompile);
    TXmlNode releasePropLink("Link");
    releasePropLink.Add(TXmlNode("EnableCOMDATFolding").Text("true"));
    releasePropLink.Add(TXmlNode("OptimizeReferences").Text("true"));
    releasePropLink.Add(TXmlNode("GenerateDebugInformation").Text("true"));
    releasePropLink.Add(TXmlNode("AdditionalOptions").Text(linkWhole));
    releaseProp.Add(releasePropLink);
    if (proj.IsUsingCuda) {
        TXmlNode releaseCuda("CudaCompile");
        releaseCuda.Add(TXmlNode("TargetMachinePlatform").Text("64"));
        releaseCuda.Add(TXmlNode("CodeGeneration").Text("compute_89,sm_89"));
        releaseCuda.Add(TXmlNode("FastMath").Text("true"));
        releaseCuda.Add(TXmlNode("HostDebugInfo").Text("true"));
        releaseCuda.Add(TXmlNode("DebugInformationFormat").Text("ProgramDatabase"));
        releaseProp.Add(releaseCuda);
    }
    root.Add(releaseProp);

    TXmlNode hFiles("ItemGroup");
    for (const TSourceFile &sf : proj.Files) {
        if (sf.Type == ST_H || sf.Type == ST_CUH) {
            hFiles.Add(TXmlNode("ClInclude").Prop("Include", pathToSrc + sf.Name));
        }
    }
    root.Add(hFiles);

    TXmlNode cudaFiles("ItemGroup");
    for (const TSourceFile &sf : proj.Files) {
        if (sf.Type == ST_CU) {
            cudaFiles.Add(TXmlNode("CudaCompile").Prop("Include", pathToSrc + sf.Name));
        }
    }
    root.Add(cudaFiles);

    TXmlNode cppFiles("ItemGroup");
    for (const TSourceFile &sf : proj.Files) {
        if (sf.Type == ST_CPP) {
            TXmlNode f = TXmlNode("ClCompile").Prop("Include", pathToSrc + sf.Name);
            if (sf.Pch == PCH_GEN) {
                f.Add(TXmlNode("PrecompiledHeader").Prop("Condition", "'$(Configuration)|$(Platform)'=='Debug|x64'").Text("Create"));
                f.Add(TXmlNode("PrecompiledHeader").Prop("Condition", "'$(Configuration)|$(Platform)'=='Release|x64'").Text("Create"));
            } else if (sf.Pch == PCH_NONE) {
                f.Add(TXmlNode("PrecompiledHeader").Prop("Condition", "'$(Configuration)|$(Platform)'=='Debug|x64'").Text("NotUsing"));
                f.Add(TXmlNode("PrecompiledHeader").Prop("Condition", "'$(Configuration)|$(Platform)'=='Release|x64'").Text("NotUsing"));
            }
            cppFiles.Add(f);
        }
    }
    root.Add(cppFiles);

    if (!proj.DepArr.empty()) {
        TXmlNode allDeps("ItemGroup");
        for (const TFolderPath &dep : proj.DepArr) {
            TString depVcproj = "$(SolutionDir)/" + GetStringPath(dep) + "/" + GetProjectName(dep) + ".vcxproj";
            TXmlNode xref = TXmlNode("ProjectReference").Prop("Include", depVcproj);
            xref.Add(TXmlNode("Project").Text(Format(GetPathGuid(dep))));
            allDeps.Add(xref);
        }
        root.Add(allDeps);
    }

    root.Add(TXmlNode("Import").Prop("Project", "$(VCTargetsPath)\\Microsoft.Cpp.targets"));

    if (proj.IsUsingCuda) {
        TXmlNode cudaTargets = TXmlNode("ImportGroup").Prop("Label", "ExtensionTargets");
        cudaTargets.Add(TXmlNode("Import").Prop("Project", "$(VCTargetsPath)\\BuildCustomizations\\CUDA 12.6.targets"));
        root.Add(cudaTargets);
    }

    RenderXml(vcProjFile, root);
}


static void PrintVcprojFilters(const TSourceFolder &proj, const TString &vcProjFile, const TString &pathToSrc)
{
    TXmlNode root("Project");
    root.Prop("ToolsVersion", "4.0");
    root.Prop("xmlns", "http://schemas.microsoft.com/developer/msbuild/2003");

    TXmlNode hFiles("ItemGroup");
    for (const TSourceFile &sf : proj.Files) {
        if (sf.Type == ST_H || sf.Type == ST_CUH) {
            hFiles.Add(TXmlNode("ClInclude").Prop("Include", pathToSrc + sf.Name));
        }
    }
    root.Add(hFiles);

    TXmlNode cudaFiles("ItemGroup");
    for (const TSourceFile &sf : proj.Files) {
        if (sf.Type == ST_CU) {
            cudaFiles.Add(TXmlNode("CudaCompile").Prop("Include", pathToSrc + sf.Name));
        }
    }
    root.Add(cudaFiles);

    TXmlNode cppFiles("ItemGroup");
    for (const TSourceFile &sf : proj.Files) {
        if (sf.Type == ST_CPP) {
            cppFiles.Add(TXmlNode("ClCompile").Prop("Include", pathToSrc + sf.Name));
        }
    }
    root.Add(cppFiles);

    RenderXml(vcProjFile + ".filters", root);
}


static void MakeDir(const TString &root, const TFolderPath &fp)
{
    TString dir = root;
    for (const TString &folder : fp.PathArr) {
        dir += "/" + folder;
        MakeDirectory(dir.c_str());
    }
}

void GenerateVSConfig(const TVector<TSourceFolder> &projArr, const TString &slnName, const TString &targetFolder, const TString &pathToProj)
{
    MakeDir(targetFolder, TFolderPath());
    PrintSln(Sprintf("%s/%s.sln", targetFolder.c_str(), slnName.c_str()), projArr);
    TString utilPath = "$(SolutionDir)/" + pathToProj;

    for (const TSourceFolder &proj : projArr) {
        MakeDir(targetFolder, proj.Path);
        TString path = GetStringPath(proj.Path);
        TString vcProjFileName = targetFolder + "/" + path + "/" + GetProjectName(proj.Path) + ".vcxproj";
        TString srcPath;
        for (yint i = 0; i < YSize(proj.Path.PathArr); ++i) {
            srcPath += "../";
        }
        srcPath += pathToProj + path + "/";
        PrintVcproj(proj, vcProjFileName, srcPath, utilPath);
        PrintVcprojFilters(proj, vcProjFileName, srcPath);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
void GenerateCMake(const TVector<TSourceFolder> &projArr, const TString &slnName, const TString &targetFolder, const TString &pathToProj, bool allowCuda)
{
    TOFStream f(targetFolder + "/CMakeLists.txt");
    f << "cmake_minimum_required(VERSION 3.22)\n";
    //#set(CMAKE_CXX_FLAGS "-Wno-error=unused-command-line-argument")
    f << "add_compile_options(-march=native -mavxvnni)\n";
    f << "include_directories(" << pathToProj.c_str() << ")\n";
    if (allowCuda) {
        f << "project(" << slnName << " LANGUAGES CXX CUDA)\n";
        f << "include_directories(\"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}\")\n";
        //f << "set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} --ptxas-options=-v\")\n";
        f << "set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} -lineinfo --use_fast_math\")\n";
    } else {
        f << "project(" << slnName << ")\n";
    }
    f << "\n";

    for (const TSourceFolder &proj : projArr) {
        if (proj.IsUsingCuda && !allowCuda) {
            printf("Ignoring CUDA project %s\n", GetProjectName(proj.Path).c_str());
            continue;
        }
        TString projType = (proj.FType == FT_LIBRARY) ? "add_library" : "add_executable";
        TString projName = GetProjectName(proj.Path);
        f << projType << "(" << projName;
        TString pathToSrc = pathToProj + GetStringPath(proj.Path) + "/";
        for (const TSourceFile &srcFile : proj.Files) {
            if (srcFile.Type == ST_CPP || srcFile.Type == ST_CU) {
                f << " " << pathToSrc + srcFile.Name;
            }
        }
        f << ")\n";
        if (!proj.DepArr.empty()) {
            f << "target_link_libraries(" << projName;
            for (const TFolderPath &dep : proj.DepArr) {
                f << " " << GetProjectName(dep);
            }
            f << ")\n";
        }
        if (proj.IsUsingCuda) {
            f << "set_property(TARGET " << projName << " PROPERTY CUDA_SEPARABLE_COMPILATION ON)\n";
            f << "set_target_properties(" << projName << " PROPERTIES CUDA_ARCHITECTURES \"80;86;89\")\n";
        }
        f << "\n";
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: [-nocuda] %s source_path sln_dir\n", argv[0]);
        printf("  source_path - relative path to source tree\n");
        printf("  sln_dir - folder name to store build files to\n");
        return -1;
    }
    bool allowCuda = true;
    int baseArg = 1;
    if (strcmp(argv[1], "-nocuda") == 0) {
        allowCuda = false;
        baseArg = 2;
    }
    //TString slnName = "eden";
    //TString repoPath = "c:/code/eden";
    //TString projPath = "c:/code/edenSln";
    //TString pathToProj = "../eden/";
    TString slnName = argv[baseArg];
    TString repoPath = argv[baseArg];
    TString projPath = argv[baseArg + 1];
    TString pathToProj = TString("../") + repoPath + "/";
    
    TSourceFolder root;
    TVector<TSourceFolder> projArr;
    ParseSourceDir(repoPath, &root, &projArr);
    if (!root.Files.empty()) {
        printf("expected no files at the root\n");
        abort();
    }
    ApplyConfig(&projArr);
    PropagateUsingCuda(&projArr);
    MakeProgDependOnLibs(&projArr);

    MakeDirectory(projPath.c_str());
#ifdef _MSC_VER
    GenerateVSConfig(projArr, slnName, projPath, pathToProj);
#else
    GenerateCMake(projArr, slnName, projPath, pathToProj, allowCuda);
#endif
    
    printf("Ok\n");
    return 0;
}
