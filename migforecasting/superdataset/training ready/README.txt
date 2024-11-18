DatasetPrep SD.py - главный программный модуль подготовки датасетов.

dollaravg.csv - среднегодовой курс доллара (для нормализации)

inflation0.csv - среднегодовая инфляция

inflation1.csv - среднегодовая накопительная инфляция от 2008 до 2023 года (для нормализации)

avgchicken.csv, avgbeef.csv - цены за кг. курицы и говядины (для нормализации цен)

ДАТАСЕТЫ:

superdataset-00.csv - 5843 examples, next year, no pricenorm, 20 feartures without: consnewapt, theatres, museums, parks, cliniccap, schoolnum, naturesecure, foodservturnover

superdataset-01.csv - отличается от датасета 00 нормализацией на основе доллара (normbydollar)

superdataset-02.csv - отличается от датасета 00 нормализацией на основе инфляции (normbyinf)

superdataset-03.csv - отличается от датасета 00 нормализацией на основе ср. цены за кг. курицы

superdataset-04.csv - отличается от датасета 00 нормализацией на основе ср. цены за кг. говядины

superdataset-10.csv - 10336 examples, next year, no pricenorm, 18 feartures without: consnewapt, theatres, museums, parks, cliniccap, schoolnum, naturesecure, foodservturnover, invest, budincome

superdataset-11.csv - отличается от датасета 10 нормализацией на основе доллара (normbydollar)

superdataset-12.csv - отличается от датасета 10 нормализацией на основе инфляции (normbyinf)

superdataset-13.csv - отличается от датасета 10 нормализацией на основе инфляции c 2014 (normbyinf)

superdataset-20.csv - 8591 examples, отличается от датасета 10 тем, что убраны все мун. образования с населением больше 100 тыс.

superdataset-21.csv - отличается от датасета 20 нормализацией на основе инфляции c 2014 (normbyinf)

superdataset-21 (negative flow).csv - 5665 examples, отличается от датасета 21 тем, что убраны примеры с положительным сальдо.

superdataset-21 (positive flow).csv - 1393 examples, отличается от датасета 21 тем, что убраны примеры с отрицательным сальдо.

superdataset-22.csv - 3990 examples, отличается от датасета 21 очисткой выбросов (алгоритм Кирилла)

superdataset-23.csv - 6122 examples, отличается от датасета 22 очисткой выбросов на уровне 4*IQR (алгоритм Кирилла)

superdataset-23 (negative flow).csv - 4208 examples, отличается от датасета 23 тем, что убраны примеры с положительным сальдо

superdataset-23 (positive flow).csv - 988 examples, отличается от датасета 23 тем, что убраны примеры с отрицательным сальдо

superdataset-24.csv - 7445 examples, отличается от датасета 23 тем, что убраны признаки: consnewareas, funds, factoriescup (maxsaldo 854)

superdataset-24-2 (negative flow).csv - 5834 examples, отличается от датасета 24 тем, что убраны примеры с положительным сальдо (maxsaldo 1046)

superdataset-24-2 (positive flow).csv - 1611 examples, отличается от датасета 24 тем, что убраны примеры с отрицательным сальдо (maxsaldo 854)

superdataset-24 inflow.csv - 7556 examples, отличается от датасета 24 тем, что вместо сальдо (saldo) используется приток (inflow) (maxinflow 3933)

superdataset-24 outflow.csv - 7661 examples, отличается от датасета 24 тем, что вместо сальдо (saldo) используется отток (outflow) (maxoutflow 4087)

superdataset-24 balanced.csv - 3222 examples, отличается от датасета 24 тем, что используется одинаковое количество примеров с отриц. и полож. сальдо (maxsaldo 854)

superdataset-24 InOut.csv - 7556 examples, отличается от датасета 24 тем, что вместо saldo есть связанные inflow и outflow (для гибридной реализации)

superdataset-24 interreg.csv - 7434 examples, отличается от датасета 24 тем, что вместо общего сальдо используется сальдо межрегиональной миграции (maxsaldo 347)

superdataset-24 reg.csv - 7459 examples, отличается от датасета 24 тем, что вместо общего сальдо используется сальдо внутрирегиональной миграции (maxsaldo 512)

superdataset-24 internat.csv - 6255 examples, отличается от датасета 24 тем, что вместо общего сальдо используется сальдо внутрирегиональной миграции (maxsaldo 295)

superdataset-24 internat balanced.csv - 3700 examples, отличается от датасета 24 internat тем, что использ. одинаковое количество примеров с отриц. и полож. сальдо (maxsaldo 294)

superdataset-24 interreg balanced.csv - 3216 examples, отличается от датасета 24 interreg тем, что использ. одинаковое количество примеров с отриц. и полож. сальдо (maxsaldo 347)

superdataset-24 reg balanced.csv - 2644 examples, отличается от датасета 24 reg тем, что использ. одинаковое количество примеров с отриц. и полож. сальдо (maxsaldo 512)

superdataset-25.csv - 4667 examples, отличается от датасета 24 тем, что добавлен признак goodcompanies

superdataset-26.csv - 2866 examples, отличается от датасета 24 тем, что добавлен признак badcompanies

superdataset-27.csv - 5795 examples, отличается от датасета 24 тем, что добавлен признак visiblecompanies

superdataset-28.csv - 4368 examples, отличается от датасета 24 тем, что добавлен признак goodcompincome

superdataset-31.csv - 894 examples, отличается от датасета 13 тем, что в датасет включены только города с населнием свыше 100 тыс.

superdataset-40.csv - 1925 examples, только образования меньше 100 тыс. и на основе ценностно-ориентированных признаков: 'foodseats', 'sportsvenue', 'servicesnum', 'museums', 'parks', 'theatres'

superdataset-41.csv - 1897 examples, отличается от датасета 40 тем, что добавлены новые ценностно-ориентированные признаки: 'library', 'cultureorg' and 'musartschool'

superdataset-41 (positive|negative).csv - отличается от датасета 41 тем, что датасет разбит на положительное и отрицательное сальдо

superdataset-41-1 (positive|negative).csv - отличается от датасета 41 тем, что датасет разбит на положительное и отрицательное сальдо, а также убраные признаки 'museums', 'parks'

superdataset-42.csv - 775 examples, отличается от датасета 41 тем, что включены только большие города

superdataset-43.csv - 13611 examples, отличается от датасета 41-1 тем, что убран признак 'theatres' и использован IQR*4 (все поселения меньше 100 тыс.)

superdataset-43 (negative flow).csv - 8640 examples, отличается от датасета 43 тем, что датасет разбит на положительное и отрицательное сальдо

superdataset-43 (positive flow).csv - 2309 examples, отличается от датасета 43 тем, что датасет разбит на положительное и отрицательное сальдо

superdataset-44.csv - 1350 examples, отличается от датасета 43 тем, что включены только большие города