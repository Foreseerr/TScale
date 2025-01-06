#pragma once
#include "bin_saver.h"


template<class T>
inline void SerializeMem(EIODirection ioDir, TVector<ui8> *data, T &c)
{
    if (IBinSaver::HasTrivialSerializer(&c)) {
        if (ioDir == IO_READ) {
            Y_ASSERT(data->size() == sizeof(T));
            c = *reinterpret_cast<T *>(data->data());
        } else {
            data->yresize(sizeof(T));
            *reinterpret_cast<T *>(data->data()) = c;
        }
    } else {
        TMemStream f(data);
        {
            TBufferedStream bufIO(ioDir, f);
            IBinSaver bs(bufIO);
            bs.Add(&c);
        }
        f.Swap(data);
    }
}
