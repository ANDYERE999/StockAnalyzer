#include "DateDialog.h"

DateDialog::DateDialog(QWidget* parent) : QDialog(parent)
{
    startDateEdit = new QDateEdit(this);
    endDateEdit = new QDateEdit(this);
    textInput = new QLineEdit(this);
    okButton = new QPushButton(tr("È·¶¨"), this);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(startDateEdit);
    layout->addWidget(endDateEdit);
    layout->addWidget(textInput);
    layout->addWidget(okButton);

    connect(okButton, &QPushButton::clicked, this, &QDialog::accept);

    setLayout(layout);
}

QDate DateDialog::getStartDate() const
{
    return startDateEdit->date();
}

QDate DateDialog::getEndDate() const
{
    return endDateEdit->date();
}

QString DateDialog::getText() const
{
    return textInput->text();
}
