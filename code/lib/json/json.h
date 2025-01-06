#pragma once
#include <util/string.h>


namespace NJson
{

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TElement
{
    // name -> value, name can be empty (for root, arrays)
    enum {
        NONE,
        ARRAY,
        OBJECT,
        VALUE,
        STRING
    };
    int NamePos = -1;
    int ValuePos = -1;
    int Type = NONE;
    int Next = -1;

    bool CanExpand() { return Type != VALUE; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TJson : public TThrRefBase
{
    TVector<char> Strings;
    TVector<TElement> Elements;

public:
    void Clear()
    {
        Strings.resize(0);
        Elements.resize(0);
    }

    const char *GetString(yint pos) const
    {
        if (pos < 0) {
            return "";
        }
        return Strings.data() + pos;
    }

    yint AddString(const TString &x)
    {
        if (x.empty()) {
            return -1;
        }
        yint res = YSize(Strings);
        Strings.insert(Strings.end(), x.begin(), x.end());
        Strings.push_back(0);
        return res;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
TIntrusivePtr<TJson> ParseJson(const TVector<char> &str);
TVector<char> Render(TIntrusivePtr<TJson> json);
TString RenderString(TIntrusivePtr<TJson> json);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TJsonIterator
{
    TIntrusivePtr<TJson> Json;
    yint ElementId = 0;

public:
    TJsonIterator(TIntrusivePtr<TJson> p, yint id = 0) : Json(p), ElementId(id)
    {
        Y_VERIFY(p.Get());
    }

    bool IsValid() const
    {
        return ElementId >= 0;
    }

    void Next()
    {
        ElementId = Json->Elements[ElementId].Next;
    }

    bool IsArray() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return elem.Type == TElement::ARRAY;
    }

    bool IsObject() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return elem.Type == TElement::OBJECT;
    }

    TJsonIterator Expand() const
    {
        Y_VERIFY(Json->Elements[ElementId].CanExpand());
        return TJsonIterator(Json, ElementId + 1);
    }

    const char *GetName() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return Json->GetString(elem.NamePos);
    }

    bool IsValue() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return elem.ValuePos >= 0;
    }

    const char *GetValue() const
    {
        const TElement &elem = Json->Elements[ElementId];
        return Json->GetString(elem.ValuePos);
    }

    bool GetBoolValue() const
    {
        return ToLower(GetValue()) == "true";
    }

    double GetFloatValue() const
    {
        return atof(GetValue());
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TJsonWriter : public TNonCopyable
{
    TIntrusivePtr<TJson> Json;
    TVector<yint> ElementStack;

    yint AddNode(bool doPush)
    {
        yint elemId = YSize(Json->Elements);
        if (!ElementStack.empty()) {
            if (ElementStack.back() >= 0) {
                TElement &prev = Json->Elements[ElementStack.back()];
                prev.Next = elemId;
            }
            ElementStack.back() = elemId;
        }
        Json->Elements.push_back();
        if (doPush) {
            ElementStack.push_back(-1);
        }
        return elemId;
    }

    TElement &AddElement(bool doPush, const TString &name, yint tt)
    {
        TElement &add = Json->Elements[AddNode(doPush)];
        add.NamePos = Json->AddString(name);
        add.Type = tt;
        return add;
    }
public:
    TJsonWriter(TIntrusivePtr<TJson> json) : Json(json)
    {
        Json->Clear();
    }

    void AddValue(const TString &name, const TString &value)
    {
        TElement &add = AddElement(false, name, TElement::VALUE);
        add.ValuePos = Json->AddString(value);
    }

    void AddString(const TString &name, const TString &value)
    {
        TElement &add = AddElement(false, name, TElement::STRING);
        add.ValuePos = Json->AddString(value);
    }

    void AddBool(const TString &name, bool x)
    {
        TElement &add = AddElement(false, name, TElement::VALUE);
        add.ValuePos = Json->AddString(x ? "True" : "False");
    }

    void AddFloat(const TString &name, double x)
    {
        TElement &add = AddElement(false, name, TElement::VALUE);
        add.ValuePos = Json->AddString(Sprintf("%g", x));
    }

    void AddArray(const TString &name)
    {
        AddElement(true, name, TElement::ARRAY);
    }

    void AddObject(const TString &name)
    {
        AddElement(true, name, TElement::OBJECT);
    }

    void Finish()
    {
        Y_VERIFY(!ElementStack.empty());
        ElementStack.pop_back();
    }
};


void Test();
}

